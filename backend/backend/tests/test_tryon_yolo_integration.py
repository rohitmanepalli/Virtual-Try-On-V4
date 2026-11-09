"""
YOLO person validation in /api/tryon.

YOLO is a presence gate only: it must reject captures with no person before any Gemini call,
never alter the capture handed to the rest of the pipeline, never leak into the response,
and never take the server down when the detector itself misbehaves.
"""
import asyncio
import base64
import io
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image

from app.models import ClothAnalysis, SilhouetteVerificationResult, TryOnRequest, TryOnResponse
from app.routers import tryon
from app.services.yolo_service import PersonDetection, YOLODetectionResult, YOLOService

W, H = 384, 512


def studio_img() -> np.ndarray:
    img = np.full((H, W, 3), 238, dtype=np.uint8)
    cv2.rectangle(img, (int(W * 0.35), int(H * 0.15)), (int(W * 0.65), int(H * 0.95)), (90, 60, 40), -1)
    return img


def _jpeg(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _data_uri(arr: np.ndarray) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(arr)).decode()


def _det(x1, y1, x2, y2, conf) -> PersonDetection:
    return PersonDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, area=(x2 - x1) * (y2 - y1))


def _result(dets, w=W, h=H) -> YOLODetectionResult:
    return YOLODetectionResult(
        detections=dets, primary_person=YOLOService.select_primary_person(dets),
        count=len(dets), image_width=w, image_height=h,
    )


def fake_service(result=None, error=None) -> MagicMock:
    svc = MagicMock()
    if error is not None:
        svc.detect_person.side_effect = error
    else:
        svc.detect_person.return_value = result
    return svc


CLOTH = ClothAnalysis(
    category="shirt", garment_name="Test Shirt", fabric_type="cotton", structure_level="soft_drape",
    is_outerwear=False, sleeve_behavior="natural", tryon_guidelines="fit naturally",
)
PASSING = SilhouetteVerificationResult(score=0.9, passed=True, threshold=0.6, details={})
USER_CAPTURE = studio_img()
GEMINI_OUTPUT = _jpeg(studio_img())
ONE_PERSON = _result([_det(130, 70, 250, 490, 0.91)])


class TryOnYOLOIntegrationTest(unittest.TestCase):
    def _run(self, service=None, enabled=True, user_image=None):
        """Runs /api/tryon with Gemini and heavy MediaPipe work mocked. Returns (response|HTTPException, gemini_calls)."""
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return GEMINI_OUTPUT, 0.97, CLOTH

        get_service = MagicMock(return_value=service) if service is not None else \
            MagicMock(side_effect=AssertionError("YOLO must not be touched when validation is disabled"))
        req = TryOnRequest(
            user_image=user_image or _data_uri(USER_CAPTURE),
            dress_image=_data_uri(studio_img()),
            cloth_analysis=CLOTH,
        )
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", enabled), \
             patch.object(tryon, "_get_yolo_service", get_service), \
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
            try:
                return asyncio.run(tryon.virtual_tryon(req)), calls
            except HTTPException as exc:
                return exc, calls

    def _assert_generated_normally(self, resp, calls):
        self.assertIsInstance(resp, TryOnResponse)
        self.assertTrue(resp.success)
        self.assertEqual(len(calls), 1)

    # ── enabled + person ──────────────────────────────────────────────────
    def test_person_detected_continues_to_gemini(self):
        svc = fake_service(ONE_PERSON)
        resp, calls = self._run(svc)
        self._assert_generated_normally(resp, calls)
        svc.detect_person.assert_called_once()

    def test_capture_reaches_gemini_unmodified(self):
        """YOLO runs on a copy; the image Gemini receives is pixel-identical to the decoded capture."""
        resp, calls = self._run(fake_service(ONE_PERSON))
        self._assert_generated_normally(resp, calls)
        decoded = np.array(tryon.clean_and_decode_base64(_data_uri(USER_CAPTURE)))
        self.assertTrue(np.array_equal(np.array(calls[0]["user_img"]), decoded))

    # ── enabled + no person ───────────────────────────────────────────────
    def test_no_person_rejected_before_gemini(self):
        with patch.object(tryon, "analyze_user_appearance") as appearance:
            resp, calls = self._run(fake_service(_result([])))
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.detail, tryon.NO_PERSON_MESSAGE)
        self.assertNotIn("yolo", resp.detail.lower())
        self.assertEqual(calls, [])
        appearance.assert_not_called()

    # ── multiple people ───────────────────────────────────────────────────
    def test_multiple_people_primary_selection_is_deterministic(self):
        foreground = _det(100, 40, 300, 510, 0.88)
        background = [_det(10, 200, 60, 330, 0.95), _det(320, 180, 380, 320, 0.60)]
        dets = [foreground, *background]
        seen = set()
        for seed in range(5):
            shuffled = dets[:]
            random.Random(seed).shuffle(shuffled)
            with patch.object(tryon, "YOLO_VALIDATION_ENABLED", True), \
                 patch.object(tryon, "_get_yolo_service", return_value=fake_service(_result(shuffled))):
                meta = tryon._validate_person(Image.fromarray(USER_CAPTURE))
            seen.add(tuple(meta["bbox"]))
            self.assertEqual(meta["person_count"], 3)
        self.assertEqual(seen, {(100, 40, 300, 510)})

    def test_multiple_people_does_not_crop_or_alter_capture(self):
        dets = [_det(100, 40, 300, 510, 0.88), _det(10, 200, 60, 330, 0.95)]
        resp, calls = self._run(fake_service(_result(dets)))
        self._assert_generated_normally(resp, calls)
        self.assertEqual(calls[0]["user_img"].size, (W, H))
        decoded = np.array(tryon.clean_and_decode_base64(_data_uri(USER_CAPTURE)))
        self.assertTrue(np.array_equal(np.array(calls[0]["user_img"]), decoded))

    def test_primary_metadata(self):
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", True), \
             patch.object(tryon, "_get_yolo_service", return_value=fake_service(ONE_PERSON)):
            meta = tryon._validate_person(Image.fromarray(USER_CAPTURE))
        self.assertEqual(meta["bbox"], [130, 70, 250, 490])
        self.assertEqual(meta["confidence"], 0.91)
        self.assertEqual((meta["image_width"], meta["image_height"]), (W, H))
        self.assertEqual(meta["bbox_normalized"], [round(130 / W, 4), round(70 / H, 4), round(250 / W, 4), round(490 / H, 4)])

    # ── disabled ──────────────────────────────────────────────────────────
    def test_disabled_runs_existing_pipeline_without_touching_yolo(self):
        resp, calls = self._run(service=None, enabled=False)  # _get_yolo_service raises if called
        self._assert_generated_normally(resp, calls)

    # ── detector failures fail open ───────────────────────────────────────
    def test_inference_exception_falls_back_to_existing_pipeline(self):
        resp, calls = self._run(fake_service(error=RuntimeError("CUDA out of memory")))
        self._assert_generated_normally(resp, calls)

    def test_model_load_failure_falls_back_to_existing_pipeline(self):
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", True), \
             patch.object(tryon, "_get_yolo_service", side_effect=ImportError("No module named 'cv2'")):
            self.assertIsNone(tryon._validate_person(Image.fromarray(USER_CAPTURE)))

    def test_malformed_detection_falls_back_to_existing_pipeline(self):
        bad = YOLODetectionResult(detections=[_det(0, 0, 10, 10, 0.9)], primary_person=_det(50, 50, 10, 10, float("nan")),
                                  count=1, image_width=W, image_height=H)
        resp, calls = self._run(fake_service(bad))
        self._assert_generated_normally(resp, calls)
        resp, calls = self._run(fake_service(SimpleNamespace(detections=None, primary_person=None)))
        self._assert_generated_normally(resp, calls)

    # ── invalid input ─────────────────────────────────────────────────────
    def test_invalid_image_is_a_clean_400(self):
        svc = fake_service(ONE_PERSON)
        resp, calls = self._run(svc, user_image="data:image/jpeg;base64,bm90IGFuIGltYWdl")
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(calls, [])
        svc.detect_person.assert_not_called()

    def test_truncated_image_is_a_clean_400(self):
        """A valid header with corrupt pixel data must fail at decode, not mid-pipeline as a 500."""
        truncated = _jpeg(USER_CAPTURE)[:600]
        svc = fake_service(ONE_PERSON)
        resp, calls = self._run(svc, user_image="data:image/jpeg;base64," + base64.b64encode(truncated).decode())
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(calls, [])
        svc.detect_person.assert_not_called()

    # ── output hygiene ────────────────────────────────────────────────────
    def test_output_is_exact_gemini_image_without_overlays(self):
        resp, calls = self._run(fake_service(_result([_det(100, 40, 300, 510, 0.88), _det(10, 200, 60, 330, 0.95)])))
        self._assert_generated_normally(resp, calls)
        served = base64.b64decode(resp.result_image.split(",", 1)[1])
        self.assertEqual(served, GEMINI_OUTPUT)  # byte-identical: nothing drawn or composited
        body = resp.model_dump()
        self.assertEqual(set(body), set(TryOnResponse.model_fields))
        self.assertNotIn("bbox", str({k: v for k, v in body.items() if k != "result_image"}))


class RealYOLOTryOnTest(unittest.TestCase):
    """Runs the real YOLOv8n weights through the router gate (no Gemini)."""

    @classmethod
    def setUpClass(cls):
        ASSETS = os.path.join(os.path.dirname(__file__), "assets")
        cls.bus = Image.open(os.path.join(ASSETS, "bus.jpg")).convert("RGB")

    def _validate(self, img):
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", True):
            return tryon._validate_person(img)

    def test_real_people_pass_and_capture_is_untouched(self):
        before = np.array(self.bus).copy()
        meta = self._validate(self.bus)
        self.assertGreaterEqual(meta["person_count"], 2)
        again = self._validate(self.bus)
        meta.pop("inference_ms"), again.pop("inference_ms")
        self.assertEqual(meta, again)
        self.assertTrue(np.array_equal(np.array(self.bus), before))

    def test_real_empty_frame_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            self._validate(Image.new("RGB", (640, 480), (200, 200, 200)))
        self.assertEqual(ctx.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
