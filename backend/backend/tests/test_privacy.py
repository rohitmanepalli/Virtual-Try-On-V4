"""
Privacy guarantees: permitted business data is saved, and no visitor or generated image reaches disk, the database or a log,
including when a request fails. Runs the real /api/tryon route with Gemini and validation replaced.
"""
import base64
import io
import os
import sqlite3
import time
import unittest
from unittest.mock import patch

import numpy as np
import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import main as app_main
from app.models import ClothAnalysis, SilhouetteVerificationResult
from app.routers import tryon
from app.services import app_log, capture_cache, download_store, gemini_client, image_utils, privacy, visitor_store
from app.services.session_manager import session_manager

CLOTH = ClothAnalysis(category="kurta", garment_name="Test Kurta", fabric_type="Cotton", structure_level="semi_structured",
                      is_outerwear=False, sleeve_behavior="Natural", tryon_guidelines="Fit")
PASSING = SilhouetteVerificationResult(score=0.9, passed=True, threshold=0.6, details={})
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n", b"BM6", b"RIFF", b"GIF8")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def _png(colour=(120, 90, 60), size=(90, 120)):
    buf = io.BytesIO()
    Image.fromarray(np.full((size[1], size[0], 3), colour, dtype=np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _uri(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode()


CAPTURE, GARMENT = _png(), _png((10, 200, 30))
CAPTURE_B64 = base64.b64encode(CAPTURE).decode()


def _generated():
    return _png((240, 240, 240), (300, 400))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Runs from an empty folder with a fresh database, so any file the app writes is visible."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(visitor_store, "VISITOR_DB_PATH", str(tmp_path / "data" / "visitors.db"))
    monkeypatch.setattr(visitor_store, "VISITOR_DATA_ENABLED", True)
    monkeypatch.setenv("VISITOR_DATA_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(app_log, "APP_LOG", True)
    monkeypatch.setattr(app_log, "APP_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(app_log, "APP_LOG_PERSONAL_DATA", False)
    download_store.clear()
    yield tmp_path
    download_store.clear()
    for session_id in list(session_manager._sessions):
        session_manager.purge_session(session_id)


@pytest.fixture
def client():
    return TestClient(app_main.app)  # No lifespan: the sweeper thread is tested directly


def _patched(fail=None):
    def generate(**_):
        if fail:
            raise fail
        return _generated(), 0.95, CLOTH
    return [
        patch.object(tryon, "_validate_person", return_value=None), patch.object(tryon, "_validate_body"),
        patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im),
        patch.object(tryon, "analyze_user_appearance", return_value=""), patch.object(tryon, "analyze_cloth", return_value=CLOTH),
        patch.object(tryon, "_capture_has_face", return_value=True), patch.object(tryon, "_face_score", return_value=0.9),
        patch.object(tryon, "generate_tryon_image", side_effect=generate),
        patch.object(tryon, "_is_studio_render", return_value=True),
        patch.object(tryon.silhouette_verifier, "verify", return_value=PASSING),
        patch.object(tryon.body_profile_service, "extract_single_frame", return_value=None),
        patch.object(tryon.body_profile_service, "extract_profile", side_effect=RuntimeError("n/a")),
        patch.object(tryon.body_profile_service, "generate_face_and_hands_mask", return_value=(None, np.zeros((4, 4), np.float32))),
    ]


def _tryon(client, session="sess_privacy_1", fail=None, garment="kurta1"):
    patches = _patched(fail)
    for p in patches:
        p.start()
    try:
        return client.post("/api/tryon", json={"user_image": _uri(CAPTURE), "dress_image": _uri(GARMENT), "session_id": session,
                                               "selected_size": "L", "garment_id": garment})
    finally:
        for p in patches:
            p.stop()


def _image_files(root):
    """Every file under root that is an image by name or by content."""
    found = []
    for folder, _, names in os.walk(root):
        for name in names:
            path = os.path.join(folder, name)
            with open(path, "rb") as fh:
                head = fh.read(8)
            if name.lower().endswith(IMAGE_EXTENSIONS) or head.startswith(IMAGE_MAGIC):
                found.append(path)
    return found


def _db_blobs(path):
    """(table, column, value) of every stored value."""
    db = sqlite3.connect(path)
    values = []
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
        cursor = db.execute(f"SELECT * FROM {table}")
        columns = [c[0] for c in cursor.description]
        for row in cursor.fetchall():
            values.extend((table, c, v) for c, v in zip(columns, row))
    db.close()
    return values


class TestNoImageOnDisk:
    def test_successful_tryon_writes_no_image_anywhere(self, workspace, client):
        response = _tryon(client)
        assert response.status_code == 200 and response.json()["success"]
        assert _image_files(workspace) == []
        assert not os.path.exists(workspace / "temp_images")

    @pytest.mark.parametrize("failure", [RuntimeError("gemini exploded"), HTTPException(status_code=502, detail="no face")])
    def test_failed_tryon_writes_no_image_anywhere(self, workspace, client, failure):
        assert _tryon(client, fail=failure).status_code in (500, 502)
        assert _image_files(workspace) == []

    def test_rejected_capture_writes_no_image_and_stays_out_of_memory(self, workspace, client):
        patches = _patched()
        patches[0] = patch.object(tryon, "_validate_person", side_effect=HTTPException(status_code=422, detail="no person"))
        for p in patches:
            p.start()
        try:
            response = client.post("/api/tryon", json={"user_image": _uri(CAPTURE), "dress_image": _uri(GARMENT), "session_id": "sess_r"})
        finally:
            for p in patches:
                p.stop()
        assert response.status_code == 422
        assert _image_files(workspace) == []
        assert capture_cache.get(capture_cache.key_for(Image.open(io.BytesIO(CAPTURE)))) is None

    def test_photo_saving_aids_default_to_off(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "app", "config.py"), encoding="utf-8") as fh:
            source = fh.read()
        assert 'os.getenv("CALIBRATION_CAPTURE_SAVE", "false")' in source
        assert 'os.getenv("MEASUREMENT_CORRECTION_LOG", "false")' in source

    def test_no_app_code_writes_image_files(self):
        """Any new image write under app/ has to be to memory."""
        offenders = []
        root = os.path.join(os.path.dirname(__file__), "..", "app")
        for folder, _, names in os.walk(root):
            for name in names:
                if not name.endswith(".py"):
                    continue
                with open(os.path.join(folder, name), encoding="utf-8") as fh:
                    for number, line in enumerate(fh, 1):
                        if "cv2.imwrite" in line or "NamedTemporaryFile" in line or "mkstemp" in line or ("open(" in line and '"wb"' in line):
                            offenders.append(f"{name}:{number}")
        assert offenders == []

    def test_legacy_copies_are_removed_but_nothing_else(self, workspace):
        os.makedirs("temp_images")
        with open("temp_images/old.jpg", "wb") as fh:
            fh.write(b"x")
        with open("temp_images/keep.txt", "w") as fh:
            fh.write("x")
        assert privacy.remove_legacy_temp_images() == 1
        assert os.listdir("temp_images") == ["keep.txt"]

    def test_legacy_folder_is_removed_when_empty(self, workspace):
        os.makedirs("temp_images")
        with open("temp_images/old.jpg", "wb") as fh:
            fh.write(b"x")
        privacy.remove_legacy_temp_images()
        assert not os.path.exists("temp_images")


class TestQrDownload:
    def test_download_works_from_memory_and_is_not_cached_by_the_phone(self, workspace, client):
        download_store.put("abc", b"jpeg-bytes", "sess_a")
        assert "abc" in client.get("/download/abc").text
        image = client.get("/api/image/abc?download=true")
        assert image.content == b"jpeg-bytes" and image.headers["cache-control"] == "no-store"
        assert "attachment" in image.headers["content-disposition"]

    def test_unknown_or_expired_is_404(self, workspace, client, monkeypatch):
        assert client.get("/api/image/none").status_code == 404
        monkeypatch.setattr(download_store, "DOWNLOAD_TTL_SECONDS", 0)
        download_store.put("gone", b"x")
        assert client.get("/download/gone").status_code == 404

    def test_session_end_leaves_only_a_short_grace(self, workspace, monkeypatch):
        download_store.put("mine", b"x", "sess_a")
        download_store.put("theirs", b"y", "sess_b")
        monkeypatch.setattr(download_store, "DOWNLOAD_GRACE_SECONDS", 0)
        download_store.end_session("sess_a")
        assert download_store.get("mine") is None and download_store.get("theirs") == b"y"

    def test_store_is_bounded(self, workspace):
        for i in range(download_store.MAX_COPIES + 2):
            download_store.put(str(i), b"x")
        assert download_store.get("0") is None

    def test_tryon_qr_points_at_a_copy_in_memory(self, workspace, client):
        qr_targets = []
        with patch.object(tryon, "generate_qr_data_uri", side_effect=lambda url: qr_targets.append(url) or "qr"):
            assert _tryon(client).status_code == 200
        image_id = qr_targets[0].rsplit("/", 1)[1]
        served = client.get(f"/api/image/{image_id}")
        assert served.status_code == 200 and served.content.startswith(b"\xff\xd8\xff")
        assert _image_files(workspace) == []


class TestVisitorRecords:
    def _details(self, client, **overrides):
        body = {"session_id": "sess_privacy_1", "consent": True, "consent_version": "v1", "name": "Asha Rao",
                "phone": "98765 43210", "gender": "female", **overrides}
        return client.post("/api/visitors", json=body)

    def test_tryon_is_recorded_anonymously_without_consent(self, workspace, client):
        _tryon(client)
        row = visitor_store.export_rows()[0]
        assert (row["garment_id"], row["clothing_category"], row["selected_size"], row["outcome"]) == ("kurta1", "kurta", "L", "served")
        assert row["created_at"] and row["visit_started_at"] and row["duration_s"] is not None
        assert row["visitor_id"] is None and row["name"] == "" and row["phone"] == ""

    def test_failed_tryon_is_recorded_too(self, workspace, client):
        _tryon(client, fail=HTTPException(status_code=502, detail="no face"))
        assert visitor_store.export_rows()[0]["outcome"] != "served"

    def test_consented_details_are_saved_and_linked(self, workspace, client):
        _tryon(client)
        assert self._details(client).json() == {"saved": True}
        row = visitor_store.export_rows()[0]
        assert (row["name"], row["phone"], row["gender"], row["consent_version"]) == ("Asha Rao", "919876543210", "female", "v1")
        assert row["visitor_id"] and row["garment_id"] == "kurta1"

    def test_nothing_saved_without_consent(self, workspace, client):
        _tryon(client)
        assert self._details(client, consent=False).json() == {"saved": False}
        values = _db_blobs(visitor_store.VISITOR_DB_PATH)
        assert not [v for t, _, v in values if t == "visitors"]

    def test_returning_visitor_is_matched_across_visits_by_phone(self, workspace, client):
        _tryon(client, session="sess_visit_1")
        self._details(client, session_id="sess_visit_1")
        _tryon(client, session="sess_visit_2", garment="sherwani1")
        self._details(client, session_id="sess_visit_2", name=None, phone="+91 98765 43210", gender=None)
        rows = visitor_store.export_rows()
        assert len({r["visitor_id"] for r in rows}) == 1 and len({r["session_id"] for r in rows}) == 2
        assert rows[1]["name"] == "Asha Rao"  # A blank field keeps the earlier value

    def test_different_people_stay_separate(self, workspace, client):
        _tryon(client, session="sess_a")
        self._details(client, session_id="sess_a")
        _tryon(client, session="sess_b")
        self._details(client, session_id="sess_b", name="Ravi", phone="9000000000", gender="male")
        rows = {r["session_id"]: r for r in visitor_store.export_rows()}
        assert rows["sess_a"]["visitor_id"] != rows["sess_b"]["visitor_id"]
        assert (rows["sess_a"]["name"], rows["sess_b"]["name"]) == ("Asha Rao", "Ravi")

    def test_name_and_phone_are_not_readable_in_the_database_file(self, workspace, client):
        self._details(client)
        with open(visitor_store.VISITOR_DB_PATH, "rb") as fh:
            raw = fh.read()
        for secret in (b"Asha", b"9876543210"):
            assert secret not in raw

    def test_details_are_refused_rather_than_stored_readable_without_the_key(self, workspace, client, monkeypatch):
        monkeypatch.delenv("VISITOR_DATA_KEY")
        assert self._details(client).status_code == 503
        assert not os.path.exists(visitor_store.VISITOR_DB_PATH) or not [
            v for t, _, v in _db_blobs(visitor_store.VISITOR_DB_PATH) if t == "visitors"]

    def test_invalid_phone_is_rejected(self, workspace, client):
        assert self._details(client, phone="not a number").status_code == 422

    def test_an_image_cannot_be_submitted_as_a_detail(self, workspace, client):
        assert self._details(client, name=_uri(CAPTURE)).status_code == 422
        assert self._details(client, image=_uri(CAPTURE)).status_code == 422

    def test_inferred_gender_is_not_accepted_as_free_text(self, workspace, client):
        assert self._details(client, gender="Male, Medium warm-brown skin").status_code == 422

    def test_database_holds_only_text_and_numbers_and_no_image(self, workspace, client):
        _tryon(client)
        self._details(client)
        values = _db_blobs(visitor_store.VISITOR_DB_PATH)
        assert values
        for table, column, value in values:
            assert not isinstance(value, (bytes, bytearray)), f"{table}.{column} holds binary"
            assert value is None or len(str(value)) < 400, f"{table}.{column} is long enough to hide an image"
            assert CAPTURE_B64[:40] not in str(value) and "data:image" not in str(value)

    def test_schema_has_no_binary_column(self, workspace, client):
        _tryon(client)
        db = sqlite3.connect(visitor_store.VISITOR_DB_PATH)
        for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
            for column in db.execute(f"PRAGMA table_info({table})").fetchall():
                assert column[2] in ("TEXT", "INTEGER", "REAL"), f"{table}.{column[1]}"
        db.close()

    def test_no_route_reads_visitor_data_back(self):
        paths = app_main.app.openapi()["paths"]
        assert list(paths["/api/visitors"]) == ["post"]
        assert [p for p in paths if "visitor" in p] == ["/api/visitors"]

    def test_deletion_request_removes_details_and_keeps_anonymous_try_ons(self, workspace, client):
        _tryon(client)
        self._details(client)
        assert visitor_store.delete_visitor_by_phone("9876543210")
        rows = visitor_store.export_rows()
        assert len(rows) == 1 and rows[0]["visitor_id"] is None and rows[0]["name"] == ""
        assert not visitor_store.delete_visitor_by_phone("9876543210")

    def test_retention_purge(self, workspace, client):
        _tryon(client)
        self._details(client)
        assert visitor_store.purge_older_than(1) == {"visitors": 0, "transactions": 0}
        db = sqlite3.connect(visitor_store.VISITOR_DB_PATH)
        db.execute("UPDATE visitors SET updated_at = '2000-01-01T00:00:00'")
        db.execute("UPDATE tryon_transactions SET created_at = '2000-01-01T00:00:00'")
        db.commit()
        db.close()
        assert visitor_store.purge_older_than(1) == {"visitors": 1, "transactions": 1}

    def test_a_database_failure_never_breaks_the_tryon(self, workspace, client, monkeypatch):
        monkeypatch.setattr(visitor_store, "VISITOR_DB_PATH", str(workspace))  # A folder: cannot be opened
        assert _tryon(client).status_code == 200


class TestLogs:
    def _log(self, workspace):
        folder = workspace / "logs"
        return "".join(p.read_text(encoding="utf-8") for p in folder.iterdir()) if folder.exists() else ""

    def _print(self, *lines):
        tee = app_log._Tee(io.StringIO())
        for line in lines:
            print(line, file=tee)

    def test_full_tryon_leaves_no_image_data_in_the_log(self, workspace, client, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "stdout", app_log._Tee(sys.stdout))
        _tryon(client)
        client.post("/api/visitors", json={"session_id": "sess_privacy_1", "consent": True, "name": "Asha Rao", "phone": "9876543210"})
        monkeypatch.undo()
        log = self._log(workspace) if (workspace / "logs").exists() else ""
        assert CAPTURE_B64[:40] not in log and "data:image" not in log

    @pytest.mark.parametrize("line", [
        "[USER APPEARANCE] Profile created and cached: Male, Medium warm-brown, glasses: dark, build: heavy, midsection: belly",
        "[COVERAGE] Capture: own sleeves full | wrists: kada | skin #6D4F39",
        "[SILHOUETTE VERIFY] Score: 0.7 | Midsection: +0.7% (1.017 -> 1.024)",
        "[FIT CHECK] Attempt 1 size L (perfect): torso 1.0111 outer 1.5495",
        "[MIDSECTION WARNING] Serving attempt 1 with midsection change -9.0%",
        "[API DEBUG] Gemini text part: The man has a beard and glasses",
    ])
    def test_descriptions_of_the_visitor_are_withheld(self, workspace, line):
        self._print(line)
        log = self._log(workspace)
        assert app_log.WITHHELD in log
        for detail in ("warm-brown", "6D4F39", "1.017", "1.0111", "-9.0", "beard"):
            assert detail not in log

    def test_body_check_is_cut_from_an_otherwise_useful_line(self, workspace):
        self._print("[TRYON] Size: L (rec: L) | Offset: +0 | Body vs L (size steps past its max): chest=0.5 | Won't fit: no")
        log = self._log(workspace)
        assert "Size: L" in log and "chest=0.5" not in log

    @pytest.mark.parametrize("line", [
        "data:image/png;base64," + CAPTURE_B64 * 3,
        "failed payload " + "A" * 500,
    ])
    def test_image_data_is_never_logged(self, workspace, line):
        self._print(line)
        log = self._log(workspace)
        assert CAPTURE_B64[:40] not in log and "AAAAAAAAAA" not in log


class TestMemoryCleanup:
    def test_sweep_clears_an_idle_session_with_no_request_arriving(self, workspace, monkeypatch):
        session = session_manager.get_or_create_session("sess_idle")
        session.set_size_render("L", "render", dress_hash="d")
        session.last_accessed_at = time.time() - 100000
        privacy.sweep_once()
        assert session_manager._sessions.get("sess_idle") is None
        assert session.size_renders_cache == {}

    def test_sweep_clears_capture_data_when_nobody_is_mid_visit(self, workspace):
        capture_cache.put("k", capture_cache.PreparedCapture(portrait=Image.new("RGB", (4, 4)), person_meta=None))
        gemini_client._USER_APPEARANCE_CACHE["h"] = "description"
        for sid in list(session_manager._sessions):
            session_manager.purge_session(sid)
        privacy.sweep_once()
        assert capture_cache.get("k") is None and gemini_client._USER_APPEARANCE_CACHE == {}

    def test_sweep_keeps_an_active_visit(self, workspace):
        capture_cache.put("k", capture_cache.PreparedCapture(portrait=Image.new("RGB", (4, 4)), person_meta=None))
        session_manager.get_or_create_session("sess_active")
        try:
            privacy.sweep_once()
            assert capture_cache.get("k") is not None
        finally:
            session_manager.purge_session("sess_active")

    def test_sessions_are_isolated(self, workspace):
        a, b = session_manager.get_or_create_session("sess_iso_a"), session_manager.get_or_create_session("sess_iso_b")
        try:
            a.set_size_render("L", "render-a", dress_hash="d")
            assert b.get_size_render("L", dress_hash="d") is None
            session_manager.purge_session("sess_iso_a")
            assert session_manager.get_session("sess_iso_a") is None and session_manager.get_session("sess_iso_b") is b
        finally:
            session_manager.purge_session("sess_iso_b")

    def test_another_visitors_render_is_not_returned(self, workspace, client):
        _tryon(client, session="sess_first")
        record = session_manager.get_session("sess_first")
        assert record.size_renders_cache
        second = session_manager.get_or_create_session("sess_second")
        assert second.size_renders_cache == {} and second.get_last_good_render() is None
        session_manager.purge_session("sess_first")
        session_manager.purge_session("sess_second")

    def test_watermark_returns_bytes_and_writes_nothing(self, workspace):
        out = image_utils.apply_watermark(_png())
        assert out.startswith(b"\xff\xd8\xff") and _image_files(workspace) == []


if __name__ == "__main__":
    unittest.main()
