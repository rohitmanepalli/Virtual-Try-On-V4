"""
Regression guards for the "raw photo" generation path.

History: Gemini intermittently returned the visitor's original capture (office background,
webcam lighting) with only the garment swapped. Nothing detected it, so it was watermarked
and served as a normal successful try-on. These tests fail if a raw-photo render can reach
the visitor as a success again.
"""
import asyncio
import base64
import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.models import ClothAnalysis, SilhouetteVerificationResult, TryOnRequest
from app.routers import tryon
from app.services.silhouette_verifier import check_studio_background

W, H = 384, 512


def _person(bg: np.ndarray) -> np.ndarray:
    img = bg.copy()
    cv2.rectangle(img, (int(W * 0.35), int(H * 0.15)), (int(W * 0.65), int(H * 0.95)), (90, 60, 40), -1)
    return img


def studio_img() -> np.ndarray:
    return _person(np.full((H, W, 3), 238, dtype=np.uint8))


def raw_photo_img() -> np.ndarray:
    """Busy, colourful room: bunting, windows, clutter along every border."""
    rng = np.random.default_rng(7)
    bg = np.full((H, W, 3), 180, dtype=np.uint8)
    for _ in range(120):
        x, y = int(rng.integers(0, W - 20)), int(rng.integers(0, H - 20))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.rectangle(bg, (x, y), (x + int(rng.integers(8, 30)), y + int(rng.integers(8, 30))), color, -1)
    return _person(bg)


def _jpeg(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _data_uri(arr: np.ndarray) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(arr)).decode()


CLOTH = ClothAnalysis(
    category="shirt", garment_name="Test Shirt", fabric_type="cotton", structure_level="soft_drape",
    is_outerwear=False, sleeve_behavior="natural", tryon_guidelines="fit naturally",
)
PASSING = SilhouetteVerificationResult(score=0.9, passed=True, threshold=0.6, details={})


class StudioBackgroundDetectorTest(unittest.TestCase):
    def test_studio_render_passes(self):
        self.assertTrue(check_studio_background(Image.fromarray(studio_img()))[0])

    def test_raw_photo_rejected(self):
        self.assertFalse(check_studio_background(Image.fromarray(raw_photo_img()))[0])


class GenerationFallbackTest(unittest.TestCase):
    def _run(self, outputs, last_good=None):
        """Runs /api/tryon with Gemini returning `outputs` in sequence; returns (response, gemini_calls)."""
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return _jpeg(outputs[len(calls) - 1]), 0.97, CLOTH

        req = TryOnRequest(
            user_image=_data_uri(raw_photo_img()),
            dress_image=_data_uri(studio_img()),
            cloth_analysis=CLOTH,
            last_good_render=last_good,
        )
        # Synthetic rectangle "people" aren't YOLO-detectable; person validation is covered in test_tryon_yolo_integration.
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", False), \
             patch.object(tryon, "generate_tryon_image", side_effect=fake_generate), \
             patch.object(tryon, "analyze_user_appearance", return_value=""), \
             patch.object(tryon, "_capture_has_face", return_value=True), \
             patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im), \
             patch.object(tryon.silhouette_verifier, "verify", return_value=PASSING), \
             patch.object(tryon.body_profile_service, "extract_single_frame", return_value=None), \
             patch.object(tryon.body_profile_service, "extract_profile", side_effect=RuntimeError("n/a")), \
             patch.object(tryon.body_profile_service, "generate_face_and_hands_mask",
                          return_value=(None, np.zeros((H, W), np.float32))), \
             patch.object(tryon, "_face_score", return_value=0.99), \
             patch.object(tryon, "apply_watermark"), \
             patch.object(tryon, "generate_qr_data_uri", return_value="qr"):
            return asyncio.run(tryon.virtual_tryon(req)), calls

    def test_raw_photo_is_regenerated_with_studio_enforcement(self):
        resp, calls = self._run([raw_photo_img(), studio_img()])
        self.assertTrue(resp.success)
        self.assertFalse(resp.fallback_used)
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0]["enforce_studio_background"])
        self.assertTrue(calls[1]["enforce_studio_background"])
        served = np.array(Image.open(io.BytesIO(base64.b64decode(resp.result_image.split(",", 1)[1]))))
        self.assertTrue(check_studio_background(Image.fromarray(served))[0])

    def test_all_raw_photos_is_an_explicit_failure(self):
        resp, calls = self._run([raw_photo_img()] * 3)
        self.assertEqual(len(calls), 3)
        self.assertFalse(resp.success)
        self.assertIsNone(resp.result_image)
        self.assertTrue(resp.verification_failed)
        self.assertIn("try again", resp.message)

    def test_all_raw_photos_falls_back_to_studio_last_good_flagged(self):
        last_good = _data_uri(studio_img())
        resp, _ = self._run([raw_photo_img()] * 3, last_good=last_good)
        self.assertTrue(resp.fallback_used)
        self.assertEqual(resp.result_image, last_good)

    def test_raw_photo_last_good_is_never_served_or_used_as_reference(self):
        resp, calls = self._run([raw_photo_img()] * 3, last_good=_data_uri(raw_photo_img()))
        self.assertFalse(resp.success)
        self.assertIsNone(resp.result_image)
        # no earlier render is ever a generation input
        self.assertNotIn("reference_render", calls[0])


if __name__ == "__main__":
    unittest.main()
