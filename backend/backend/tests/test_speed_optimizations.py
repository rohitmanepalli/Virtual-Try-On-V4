"""Guards for the try-on speed work: reusing work must never skip a check, double-charge, or outlive the session."""
import asyncio
import base64
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.models import ClothAnalysis, PurgeSessionRequest, TryOnRequest, TryOnResponse
from app.routers import tryon
from app.services import capture_cache, gemini_client, image_utils, request_timing
from app.services.body_profile import body_profile_service
from app.services.silhouette_verifier import check_studio_background

CLOTH = ClothAnalysis(category="shirt", garment_name="Shirt", fabric_type="Cotton", structure_level="semi_structured",
                      is_outerwear=False, sleeve_behavior="Natural", tryon_guidelines="Fit")


def _image(seed=0, size=(60, 80)):
    return Image.fromarray(np.full((size[1], size[0], 3), seed, dtype=np.uint8), "RGB")


def _uri(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestCaptureCache(unittest.TestCase):
    def test_key_depends_on_every_pixel_and_the_version(self):
        a, b = _image(), _image()
        self.assertEqual(capture_cache.key_for(a), capture_cache.key_for(b))
        b.putpixel((59, 79), (1, 1, 1))
        self.assertNotEqual(capture_cache.key_for(a), capture_cache.key_for(b))
        before = capture_cache.key_for(a)
        with patch.object(capture_cache, "PREPROCESS_VERSION", "other"):
            self.assertNotEqual(before, capture_cache.key_for(a))

    def test_entries_expire_with_the_session_ttl(self):
        with patch.object(capture_cache, "SESSION_TTL_SECONDS", 0):
            capture_cache.put("k", capture_cache.PreparedCapture(portrait=_image(), person_meta=None))
            self.assertIsNone(capture_cache.get("k"))

    def test_size_is_bounded(self):
        for i in range(capture_cache.MAX_ENTRIES + 3):
            capture_cache.put(f"k{i}", capture_cache.PreparedCapture(portrait=_image(), person_meta=None))
        self.assertIsNone(capture_cache.get("k0"))
        self.assertIsNotNone(capture_cache.get(f"k{capture_cache.MAX_ENTRIES + 2}"))

    def test_a_failed_validation_is_not_cached(self):
        img = _image()
        with patch.object(tryon, "_validate_person", side_effect=tryon.HTTPException(status_code=422, detail="no")):
            with self.assertRaises(tryon.HTTPException):
                tryon._prepare_capture(img)
        self.assertIsNone(capture_cache.get(capture_cache.key_for(img)))

    def test_validation_runs_once_per_capture(self):
        with patch.object(tryon, "_validate_person", return_value=None) as person, \
                patch.object(tryon, "_validate_body") as body, \
                patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im):
            first, second = tryon._prepare_capture(_image()), tryon._prepare_capture(_image())
        self.assertIs(first, second)
        self.assertEqual((person.call_count, body.call_count), (1, 1))

    def test_a_different_capture_is_validated_again(self):
        with patch.object(tryon, "_validate_person", return_value=None) as person, \
                patch.object(tryon, "_validate_body"), \
                patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im):
            tryon._prepare_capture(_image(1))
            tryon._prepare_capture(_image(2))
        self.assertEqual(person.call_count, 2)

    def test_session_purge_drops_visitor_data(self):
        capture_cache.put("k", capture_cache.PreparedCapture(portrait=_image(), person_meta=None))
        gemini_client._USER_APPEARANCE_CACHE["h"] = "description"
        gemini_client._USER_APPEARANCE_DETAILS["h"] = {}
        asyncio.run(tryon.purge_session(PurgeSessionRequest(session_id="sess_x")))
        self.assertIsNone(capture_cache.get("k"))
        self.assertEqual((gemini_client._USER_APPEARANCE_CACHE, gemini_client._USER_APPEARANCE_DETAILS), ({}, {}))


class TestNoDuplicateGeneration(unittest.TestCase):
    def test_same_session_requests_run_one_at_a_time(self):
        running, overlaps, lock = [0], [], threading.Lock()

        def slow(_data):
            with lock:
                running[0] += 1
                overlaps.append(running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1
            return TryOnResponse(success=True, result_image=None, fit_state="standard", cloth_analysis=CLOTH)

        req = TryOnRequest(user_image="x", dress_image="y", session_id="sess_same")

        async def both():
            await asyncio.gather(tryon.virtual_tryon(req), tryon.virtual_tryon(req), tryon.virtual_tryon(req))
        with patch.object(tryon, "_virtual_tryon", side_effect=slow):
            asyncio.run(both())
        self.assertEqual(max(overlaps), 1)

    def test_other_sessions_are_not_held_up(self):
        release, overlap = threading.Event(), threading.Event()
        active = []

        def wait(_data):
            active.append(1)
            if len(active) == 2:
                overlap.set()
            release.wait(timeout=5) if len(active) == 1 else release.set()
            return TryOnResponse(success=True, result_image=None, fit_state="standard", cloth_analysis=CLOTH)

        async def both():
            await asyncio.gather(*(tryon.virtual_tryon(TryOnRequest(user_image="x", dress_image="y", session_id=s))
                                   for s in ("sess_a", "sess_b")))
        with patch.object(tryon, "_virtual_tryon", side_effect=wait):
            asyncio.run(both())
        self.assertTrue(overlap.is_set())

    def test_timing_row_starts_after_the_wait(self):
        """A duplicate's row must show its own work, not the minute it waited for the first request."""
        rows = []
        finish = request_timing.finish

        def keep(timing, outcome, size, fit_category):
            rows.append(finish(timing, outcome, size, fit_category))
            return rows[-1]

        def slow(_data):
            time.sleep(0.2 if not rows else 0)
            return TryOnResponse(success=True, result_image=None, fit_state="standard", cloth_analysis=CLOTH)

        req = TryOnRequest(user_image="x", dress_image="y", session_id="sess_row")

        async def both():
            await asyncio.gather(tryon.virtual_tryon(req), tryon.virtual_tryon(req))
        with patch.object(tryon, "_virtual_tryon", side_effect=slow), patch.object(request_timing, "finish", keep):
            asyncio.run(both())
        self.assertLess(min(float(r["total_s"]) for r in rows), 0.1)


class TestAppearance(unittest.TestCase):
    def _client(self, text='{"gender_presentation": "x", "own_sleeves": "full"}', delay=0.0):
        def generate_content(**_):
            time.sleep(delay)
            return SimpleNamespace(text=text, usage_metadata=None)
        client = MagicMock()
        client.models.generate_content.side_effect = generate_content
        return client

    def test_concurrent_requests_share_one_call(self):
        client, img = self._client(delay=0.1), _image()
        results = []
        threads = [threading.Thread(target=lambda: results.append(gemini_client.analyze_user_appearance(img, client=client)))
                   for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(client.models.generate_content.call_count, 1)
        self.assertEqual(len(set(results)), 1)

    def test_a_failure_is_not_cached(self):
        broken = MagicMock()
        broken.models.generate_content.side_effect = RuntimeError("503")
        img = _image()
        self.assertIn("identity lock", gemini_client.analyze_user_appearance(img, client=broken))
        working = self._client()
        gemini_client.analyze_user_appearance(img, client=working)
        self.assertEqual(working.models.generate_content.call_count, 1)

    def test_cache_key_includes_the_model_and_prompt_version(self):
        img = _image()
        before = gemini_client._appearance_hash(img)
        with patch.object(gemini_client, "ANALYSIS_MODEL", "other-model"):
            self.assertNotEqual(before, gemini_client._appearance_hash(img))
        with patch.object(gemini_client, "APPEARANCE_PROMPT_VERSION", "v2"):
            self.assertNotEqual(before, gemini_client._appearance_hash(img))

    def test_cache_is_bounded_and_details_leave_with_their_entry(self):
        client = self._client()
        first = _image(0)
        gemini_client.analyze_user_appearance(first, client=client)
        for seed in range(1, gemini_client.MAX_APPEARANCE_ENTRIES + 1):
            gemini_client.analyze_user_appearance(_image(seed), client=client)
        self.assertEqual(len(gemini_client._USER_APPEARANCE_CACHE), gemini_client.MAX_APPEARANCE_ENTRIES)
        self.assertEqual(gemini_client.get_appearance_details(first), {})

    def test_client_is_shared(self):
        fake = MagicMock()
        with patch.object(gemini_client.genai, "Client", fake):
            self.assertIs(gemini_client._get_client("key"), gemini_client._get_client("key"))
            self.assertEqual(fake.call_count, 1)
            gemini_client._get_client("another-key")
            self.assertEqual(fake.call_count, 2)


class TestBackgroundPreparation(unittest.TestCase):
    def test_rejected_capture_makes_no_gemini_call(self):
        with patch.object(tryon, "_validate_person", side_effect=tryon.HTTPException(status_code=422, detail="no")), \
                patch.object(tryon, "analyze_user_appearance") as appearance:
            tryon._prepare_in_background(_image())
        appearance.assert_not_called()

    def test_prepared_coverage_and_face_are_reused_by_the_try_on_helpers(self):
        img = _image()
        with patch.object(tryon, "_validate_person", return_value=None), patch.object(tryon, "_validate_body"), \
                patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im), \
                patch.object(tryon, "analyze_user_appearance", return_value=""), \
                patch.object(tryon, "get_appearance_details", return_value={"own_sleeves": "full"}), \
                patch.object(tryon, "_read_capture_coverage", return_value="coverage") as read, \
                patch.object(tryon, "_capture_has_face", return_value=True):
            tryon._prepare_in_background(img)
            prepared = tryon._prepare_capture(img)
            self.assertEqual(tryon._coverage_for(prepared), "coverage")
        self.assertEqual(read.call_count, 1)
        self.assertTrue(prepared.has_face)

    def test_coverage_without_the_appearance_answers_is_read_again(self):
        prepared = capture_cache.PreparedCapture(portrait=_image(), person_meta=None)
        with patch.object(tryon, "get_appearance_details", return_value={}), \
                patch.object(tryon, "_read_capture_coverage", return_value="coverage") as read:
            tryon._coverage_for(prepared)
            tryon._coverage_for(prepared)
        self.assertEqual(read.call_count, 2)

    def test_unreadable_image_is_ignored(self):
        with patch.object(tryon.threading, "Thread") as thread:
            tryon.prepare_capture(tryon.PrepareCaptureRequest(image="not an image"))
        thread.assert_not_called()


class TestReuseWithinARequest(unittest.TestCase):
    def test_pose_runs_once_per_image_object(self):
        img = _image(size=(128, 128))
        with patch.object(body_profile_service, "_extract_single_frame", return_value="landmarks") as run:
            self.assertEqual(body_profile_service.extract_single_frame(img), "landmarks")
            self.assertEqual(body_profile_service.extract_single_frame(img), "landmarks")
            self.assertEqual(run.call_count, 1)
            body_profile_service.extract_single_frame(_image(size=(128, 128)))
            self.assertEqual(run.call_count, 2)

    def test_pose_is_rerun_after_an_in_place_downscale(self):
        img = _image(size=(128, 128))
        with patch.object(body_profile_service, "_extract_single_frame", return_value="landmarks") as run:
            body_profile_service.extract_single_frame(img)
            img.thumbnail((64, 64))
            body_profile_service.extract_single_frame(img)
        self.assertEqual(run.call_count, 2)

    def test_studio_check_is_memoized_and_unchanged(self):
        img = _image(255, size=(384, 512))
        first = check_studio_background(img)
        with patch("numpy.percentile", side_effect=AssertionError("recomputed")):
            self.assertEqual(check_studio_background(img), first)
        self.assertTrue(first[0])

    def test_framing_is_lossless_and_gives_the_same_pixels_with_a_decoded_image(self):
        source = Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 300, 3), dtype=np.uint8), "RGB")
        buf = io.BytesIO()
        source.save(buf, format="PNG")
        mask = np.zeros((400, 300), dtype=np.float32)
        mask[100:300, 100:200] = 1.0
        plain = Image.open(io.BytesIO(image_utils.normalize_framing(buf.getvalue(), mask)))
        reused = Image.open(io.BytesIO(image_utils.normalize_framing(buf.getvalue(), mask, decoded=source)))
        self.assertEqual(plain.format, "BMP")
        self.assertEqual(plain.size, (300, 400))
        self.assertEqual(plain.tobytes(), reused.tobytes())


class TestRequestId(unittest.TestCase):
    def test_id_is_on_the_row_and_gone_after_the_request(self):
        timing = request_timing.begin("sess_x")
        self.assertEqual(request_timing.current_id(), timing.request_id)
        row = request_timing.finish(timing, "served", "L", "perfect")
        self.assertEqual(row["request_id"], timing.request_id)
        self.assertIsNone(request_timing.current_id())

    def test_log_lines_carry_the_id(self):
        from app.services import app_log
        import tempfile
        with tempfile.TemporaryDirectory() as folder, patch.object(app_log, "APP_LOG", True), \
                patch.object(app_log, "APP_LOG_DIR", folder):
            timing = request_timing.begin("sess_x")
            print("hello", file=app_log._Tee(io.StringIO()))
            request_timing._current.set(None)
            print("outside", file=app_log._Tee(io.StringIO()))
            with open(app_log._log_path(), encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        self.assertTrue(lines[0].endswith(f"[req {timing.request_id}] hello"))
        self.assertTrue(lines[1].endswith(" outside") and "[req" not in lines[1])


if __name__ == "__main__":
    unittest.main()
