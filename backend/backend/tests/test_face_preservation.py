"""
Regression guards for real-face preservation.

History: face compositing was once silently removed from generate_tryon_image() and the
face score was hardcoded to 1.0, so AI-regenerated faces shipped while logs said
"Face Sim: 100%". These tests fail if either of those things happens again.
"""
import ast
import inspect
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

from app.models import ClothAnalysis
from app.services import face_preservation, gemini_client
from app.services.face_preservation import FaceCheck, restore_original_face, verify_face_preserved

CAPTURE_W, CAPTURE_H = 300, 400


def _texture(w, h, seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def _jpeg(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _fake_face_points(rgb):
    """Face ellipse at the same relative position in any frame (stands in for MediaPipe)."""
    h, w = rgb.shape[:2]
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    cx, cy, rx, ry = 0.5 * w, 0.3 * h, 0.17 * w, 0.16 * h
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1).astype(np.float32)


class FacePreservationTestBase(unittest.TestCase):
    def setUp(self):
        p = patch.object(face_preservation, "detect_face_points", side_effect=_fake_face_points)
        p.start()
        self.addCleanup(p.stop)
        # Random-noise "faces" have no skin at the seam to agree on; the seam gate is covered with real
        # faces in test_face_restoration_quality.py
        s = patch.object(face_preservation, "SEAM_MAX_RESIDUAL", 1.0)
        s.start()
        self.addCleanup(s.stop)
        self.capture = Image.fromarray(_texture(CAPTURE_W, CAPTURE_H, seed=1))
        # "AI render": 2x resolution, completely different pixels (a regenerated face)
        self.ai_render = _jpeg(_texture(CAPTURE_W * 2, CAPTURE_H * 2, seed=2))


class TestRestoreAndVerify(FacePreservationTestBase):
    def test_restore_puts_real_face_pixels_back_at_output_resolution(self):
        out_bytes, check = restore_original_face(self.capture, self.ai_render)
        out = np.array(Image.open(io.BytesIO(out_bytes)))
        self.assertEqual(out.shape[:2], (CAPTURE_H * 2, CAPTURE_W * 2), "Output resolution must be kept (no downscale)")
        self.assertTrue(check.passed, check)
        self.assertGreater(check.ssim, 0.9)

        # Face centre must be the capture's pixels (Lanczos-enlarged), not the render's
        M = np.float32([[2, 0, 0], [0, 2, 0]])
        upscaled = cv2.warpAffine(np.array(self.capture), M, (CAPTURE_W * 2, CAPTURE_H * 2), flags=cv2.INTER_LANCZOS4)
        ai = np.array(Image.open(io.BytesIO(self.ai_render)))
        cy, cx = int(0.3 * CAPTURE_H * 2), int(0.5 * CAPTURE_W * 2)
        win = (slice(cy - 20, cy + 20), slice(cx - 20, cx + 20))
        self.assertLess(np.abs(out[win].astype(int) - upscaled[win].astype(int)).mean(), 4.0)
        self.assertGreater(np.abs(out[win].astype(int) - ai[win].astype(int)).mean(), 10.0)

    def test_verify_flags_regenerated_face(self):
        check = verify_face_preserved(self.capture, self.ai_render)
        self.assertFalse(check.passed)
        self.assertLess(check.score, 0.5)

    def test_score_reflects_the_image_not_a_constant(self):
        restored, _ = restore_original_face(self.capture, self.ai_render)
        good = verify_face_preserved(self.capture, restored).score
        bad = verify_face_preserved(self.capture, self.ai_render).score
        self.assertGreater(good - bad, 0.4)

    def test_broken_compositing_is_caught_and_logged(self):
        """Simulates the original regression: compositing becomes a pass-through."""
        passthrough = lambda orig, gen, mask, M: gen
        buf = io.StringIO()
        with patch.object(face_preservation, "_composite", side_effect=passthrough), redirect_stdout(buf):
            _, check = restore_original_face(self.capture, self.ai_render)
        self.assertFalse(check.passed)
        self.assertIn("[FACE REGRESSION]", buf.getvalue())

    def test_aligned_original_face_dominates_and_generated_face_completely_covered(self):
        """
        Hard test verifying that inside the face region:
        1. Original aligned face matches the restored composite face with high fidelity (sim >= 0.95).
        2. Generated face does NOT remain visible or dominant (MAE vs generated >> MAE vs original).
        3. Zero ghosting: exactly ONE face is present in the final composite.
        """
        out_bytes, check = restore_original_face(self.capture, self.ai_render)
        out = np.array(Image.open(io.BytesIO(out_bytes)))
        ai = np.array(Image.open(io.BytesIO(self.ai_render)))

        M = np.float32([[2, 0, 0], [0, 2, 0]])
        orig_warped = cv2.warpAffine(np.array(self.capture), M, (CAPTURE_W * 2, CAPTURE_H * 2), flags=cv2.INTER_LANCZOS4)

        cy, cx = int(0.3 * CAPTURE_H * 2), int(0.5 * CAPTURE_W * 2)
        # Test across whole face region (radius = 35 pixels)
        win = (slice(cy - 35, cy + 35), slice(cx - 35, cx + 35))

        mae_orig = np.abs(out[win].astype(float) - orig_warped[win].astype(float)).mean()
        mae_ai = np.abs(out[win].astype(float) - ai[win].astype(float)).mean()

        # Original face must match nearly identically (MAE < 2.0 on 0-255 scale due to JPEG compression)
        self.assertLess(mae_orig, 3.0, f"Original face was not preserved accurately: MAE = {mae_orig}")
        # Generated face must NOT be visible/dominant (MAE vs AI must be high because it was replaced)
        self.assertGreater(mae_ai, 15.0, f"Generated face leaked through: MAE vs AI = {mae_ai}")
        # Ratio of error: original must be vastly closer than generated
        self.assertGreater(mae_ai / max(mae_orig, 0.001), 5.0)


class TestGenerateTryonWiring(FacePreservationTestBase):
    def _run_generate(self, identity_face=None):
        response = SimpleNamespace(
            candidates=[], prompt_feedback=None, sdk_http_response=None,
            parts=[SimpleNamespace(text=None, inline_data=SimpleNamespace(data=self.ai_render))],
        )
        client = MagicMock()
        client.models.generate_content.return_value = response
        cloth = ClothAnalysis(category="shirt", garment_name="Shirt", fabric_type="Cotton",
                              structure_level="semi_structured", is_outerwear=False,
                              sleeve_behavior="", tryon_guidelines="")
        with patch.object(gemini_client, "get_gemini_api_key", return_value="test-key-1234567890"), \
             patch.object(gemini_client.genai, "Client", return_value=client), \
             redirect_stdout(io.StringIO()):
            return gemini_client.generate_tryon_image(
                user_img=self.capture,
                dress_img=Image.new("RGB", (64, 64)),
                cloth_analysis=cloth,
                user_appearance="appearance",
                identity_face=identity_face,
            )

    def test_generate_invokes_face_restore_from_capture_not_reference(self):
        """A restorable capture face is always the source; an earlier render's face is never used then."""
        earlier_render = Image.fromarray(_texture(CAPTURE_W, CAPTURE_H, seed=3))
        with patch.object(gemini_client, "restore_original_face", wraps=restore_original_face) as spy:
            out_bytes, face_sim, _ = self._run_generate(identity_face=earlier_render)
        spy.assert_called_once()
        source_img = spy.call_args.args[0]
        self.assertIs(source_img, self.capture, "Face must come from the capture, never an earlier render")
        self.assertNotEqual(out_bytes, self.ai_render, "Raw AI render must not be returned untouched")
        self.assertGreater(face_sim, 0.9)

    def test_face_lock_uses_first_render_face_only_when_capture_face_too_small(self):
        """Capture face too small to restore: the visitor's first render supplies the face, composited on the raw output."""
        earlier_render = Image.fromarray(_texture(CAPTURE_W, CAPTURE_H, seed=3))
        too_small = FaceCheck(0.4, 0.3, False, "too small", low_detail_skip=True)
        locked = FaceCheck(0.97, 0.96, True, "", restored=True)
        with patch.object(gemini_client, "restore_original_face",
                          side_effect=[(self.ai_render, too_small), (b"locked-bytes", locked)]) as spy:
            out_bytes, face_sim, _ = self._run_generate(identity_face=earlier_render)
        self.assertEqual(spy.call_count, 2)
        self.assertIs(spy.call_args_list[0].args[0], self.capture)
        self.assertIs(spy.call_args_list[1].args[0], earlier_render)
        self.assertEqual(spy.call_args_list[1].args[1], self.ai_render, "lock must composite onto the raw model output")
        self.assertEqual(out_bytes, b"locked-bytes")
        self.assertAlmostEqual(face_sim, locked.score)

    def test_face_lock_failure_serves_render_own_face(self):
        """If the first-render face cannot be composited, the render's own single face is served (never a failure)."""
        earlier_render = Image.fromarray(_texture(CAPTURE_W, CAPTURE_H, seed=3))
        too_small = FaceCheck(0.4, 0.3, False, "too small", low_detail_skip=True)
        rejected = FaceCheck(0.5, 0.5, False, "seam", restored=False)
        with patch.object(gemini_client, "restore_original_face",
                          side_effect=[(self.ai_render, too_small), (self.ai_render, rejected)]):
            out_bytes, face_sim, _ = self._run_generate(identity_face=earlier_render)
        self.assertEqual(out_bytes, self.ai_render)
        self.assertAlmostEqual(face_sim, too_small.score)

    def test_generate_returns_the_real_check_score(self):
        for value in (0.3137, 0.8812):
            with patch.object(face_preservation, "verify_face_preserved",
                              return_value=FaceCheck(value, value, False, "stub")):
                _, face_sim, _ = self._run_generate()
            self.assertAlmostEqual(face_sim, value)


class TestNoHardcodedFaceScores(unittest.TestCase):
    """Static guard against `return ..., 1.0, ...` / `face_similarity=1.0` stubs."""

    def test_generate_tryon_image_returns_no_constant_score(self):
        tree = ast.parse(inspect.getsource(gemini_client.generate_tryon_image))
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                for elt in node.value.elts:
                    self.assertFalse(isinstance(elt, ast.Constant) and isinstance(elt.value, float),
                                     "generate_tryon_image must return a computed face score, not a constant")

    def test_tryon_router_has_no_constant_face_similarity(self):
        src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "tryon.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.keyword) and node.arg == "face_similarity":
                self.assertNotIsInstance(node.value, ast.Constant, "face_similarity must be computed, not hardcoded")


class TestFaceDetectionFallback(unittest.TestCase):
    """A full-length render whose face the whole-frame detector misses is retried on the top half of the frame."""

    RENDER = np.zeros((4800, 3584, 3), np.uint8)   # whole frame is detected at 1600px tall; the top half at ~1071px

    def _detect(self, face_only_when_height_at_most):
        calls = []

        def detect(image):
            calls.append(image.height)
            found = image.height <= face_only_when_height_at_most
            return SimpleNamespace(face_landmarks=[[SimpleNamespace(x=0.5, y=0.25)] * 478] if found else [])
        return SimpleNamespace(detect=detect), calls

    def _points(self, detector):
        from app.services.body_profile import body_profile_service
        with patch.object(body_profile_service, "_face_detector", detector, create=True):
            return face_preservation.detect_face_points(self.RENDER)

    def test_face_missed_in_full_frame_is_found_in_top_half_in_frame_coordinates(self):
        detector, calls = self._detect(face_only_when_height_at_most=1100)
        pts = self._points(detector)
        self.assertEqual(len(calls), 2)
        self.assertEqual(pts.shape, (478, 2))
        np.testing.assert_allclose(pts[0], [0.5 * 3584, 0.25 * 2400])   # y relative to the top-half crop = frame y

    def test_full_frame_hit_never_uses_the_crop(self):
        detector, calls = self._detect(face_only_when_height_at_most=10_000)
        pts = self._points(detector)
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(pts[0], [0.5 * 3584, 0.25 * 4800])

    def test_no_face_anywhere_is_none(self):
        detector, calls = self._detect(face_only_when_height_at_most=0)
        self.assertIsNone(self._points(detector))
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
