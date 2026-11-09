import io
import logging
import sys
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import client_log
from app.services import app_log


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_log, "APP_LOG", True)
    monkeypatch.setattr(app_log, "APP_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(app_log, "APP_LOG_PERSONAL_DATA", False)
    return tmp_path


def read_log(log_dir):
    return (log_dir / f"app_{datetime.now():%Y-%m-%d}.log").read_text(encoding="utf-8")


def tee_print(*args):
    console = io.StringIO()
    print(*args, file=app_log._Tee(console))
    return console.getvalue()


def test_print_reaches_both_console_and_file(log_dir):
    assert tee_print("[TRYON] Initiating generation") == "[TRYON] Initiating generation\n"
    assert "[TRYON] Initiating generation\n" in read_log(log_dir)


def test_multi_line_print_gets_one_timestamped_line_each(log_dir):
    tee_print("first\nsecond")
    lines = read_log(log_dir).splitlines()
    assert [line[20:] for line in lines] == ["first", "second"]


@pytest.mark.parametrize("label, printed", [
    ("Raw body_profile in payload", 'Raw body_profile in payload:\n{\n  "chest": 91.5\n}'),
    ("[SESSION RETRIEVED PROFILE]", '[SESSION RETRIEVED PROFILE] for sess_1:\n{\n  "chest": 91.5\n}'),
    ("[SIZE REC ERROR]", "[SIZE REC ERROR] Missing body_profile! body_profile: chest 91.5"),
    ("[SIZE REC]", "[SIZE REC] Recommended: M | Votes: [chest=91.5] | Body: pear"),
    ("[MEASUREMENT CORRECTION]", "[MEASUREMENT CORRECTION] chest=91.5"),
    ("Extracted on-the-fly BodyProfile", "[TRYON] Extracted on-the-fly BodyProfile: Sh/Hip=91.5"),
    ("Baseline fit landmarks recorded", "[TRYON] Baseline fit landmarks recorded for session s: 91.5"),
])
def test_body_data_is_withheld_from_file_but_kept_on_console(log_dir, label, printed):
    assert "91.5" in tee_print(printed)
    log = read_log(log_dir)
    assert f"{label} {app_log.WITHHELD}" in log
    assert "91.5" not in log


def test_body_data_is_written_when_explicitly_enabled(log_dir, monkeypatch):
    monkeypatch.setattr(app_log, "APP_LOG_PERSONAL_DATA", True)
    tee_print("[MEASUREMENT CORRECTION] chest=91.5")
    assert "chest=91.5" in read_log(log_dir)


def test_similar_operational_labels_are_not_withheld(log_dir):
    tee_print("[MEASUREMENT CORRECTION LOG WARNING] Could not append")
    tee_print("[SIZE RECOMMENDATION ERROR] boom")
    log = read_log(log_dir)
    assert app_log.WITHHELD not in log and "Could not append" in log and "boom" in log


def test_api_key_and_encoded_images_never_reach_the_file(log_dir):
    tee_print("[API CONFIG] Using GEMINI_API_KEY: AIzaSy...abcd | Size: M")
    tee_print("POST https://example.test/v1?key=SECRET123&x=1")
    tee_print("image: " + "QUJD" * 100)
    log = read_log(log_dir)
    assert "AIzaSy" not in log and "SECRET123" not in log and "QUJD" not in log


def test_uvicorn_colour_codes_are_stripped(log_dir):
    tee_print("\x1b[32mINFO\x1b[0m: ok")
    assert "INFO: ok" in read_log(log_dir)


def test_log_failure_does_not_break_printing(log_dir, monkeypatch):
    blocker = log_dir / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setattr(app_log, "APP_LOG_DIR", str(blocker))
    assert tee_print("still works") == "still works\n"


def test_setup_is_idempotent_and_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(app_log, "APP_LOG", False)
    app_log.setup_logging()
    assert not isinstance(sys.stdout, app_log._Tee)


def test_setup_silences_the_reloader_so_log_writes_cannot_feed_it(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(app_log, "APP_LOG", True)
    watchfiles_logger = logging.getLogger("watchfiles")
    monkeypatch.setattr(watchfiles_logger, "level", logging.NOTSET)
    app_log.setup_logging()
    assert not logging.getLogger("watchfiles.main").isEnabledFor(logging.INFO)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(client_log.router)
    return TestClient(app)


def post_entries(client, entries, client_id="abc123"):
    return client.post("/api/client-logs", json={"client_id": client_id, "entries": entries})


def test_client_entries_are_logged_in_order(client, log_dir):
    response = post_entries(client, [
        {"level": "info", "message": "click: CLOTHING", "time": "12:00:01.100"},
        {"level": "error", "message": "camera failed", "time": "12:00:02.200"},
    ])
    assert response.status_code == 204
    lines = read_log(log_dir).splitlines()
    assert lines[0].endswith("[CLIENT abc123 12:00:01.100] INFO click: CLOTHING")
    assert lines[1].endswith("[CLIENT abc123 12:00:02.200] ERROR camera failed")


def test_client_cannot_forge_log_lines(client, log_dir):
    post_entries(client, [{"level": "x\ny", "message": "a\n2026-01-01 00:00:00 forged", "time": "\nz"}], "id\nforged")
    assert len(read_log(log_dir).splitlines()) == 1


def test_client_body_data_is_withheld_with_the_client_prefix_kept(client, log_dir):
    post_entries(client, [{"message": '[BODY PROFILE EXTRACTED] {"chest": 91.5}'}])
    log = read_log(log_dir)
    assert "91.5" not in log and f"[CLIENT abc123 ] LOG [BODY PROFILE EXTRACTED] {app_log.WITHHELD}" in log


def test_oversized_batch_is_rejected(client, log_dir):
    assert post_entries(client, [{"message": "x"}] * 201).status_code == 422
    assert not list(log_dir.iterdir())


def test_client_log_stops_at_the_size_cap(client, log_dir, monkeypatch):
    post_entries(client, [{"message": "first"}])
    monkeypatch.setattr(app_log, "MAX_CLIENT_LOG_BYTES", 1)
    post_entries(client, [{"message": "second"}])
    log = read_log(log_dir)
    assert "first" in log and "second" not in log
