"""
YOLO + MediaPipe body cross-validation.

YOLO says whether there is a person and which one is primary; MediaPipe says where the body
landmarks are. A try-on may only be generated when both agree on the same, sufficiently visible
primary person, and validation must never touch pixels (face-preservation regression guard).
"""
import asyncio
import base64
import io
import os
import random
import re
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image

from app.models import ClothAnalysis, SilhouetteVerificationResult, TryOnRequest
from app.routers import tryon
from app.services import body_validation as bv_module, capture_cache
from app.services.body_profile import FrameLandmarks
from app.services.body_validation import (
    MATCH_MESSAGE,
    NO_PERSON_MESSAGE,
    REASON_BBOX_INCONSISTENT,
    REASON_BODY_CROPPED,
    REASON_DISABLED,
    REASON_LOW_VISIBILITY,
    REASON_MEDIAPIPE_UNAVAILABLE,
    REASON_NO_PERSON,
    REASON_OK,
    REASON_OTHER_PERSON,
    REASON_YOLO_MISSING,
    REASON_YOLO_UNAVAILABLE,
    VISIBILITY_MESSAGE,
    BodyValidator,
    parse_required_landmarks,
)
from app.services.yolo_service import PersonDetection, YOLODetectionResult, YOLOService

IMG = (1000, 1000)
PRIMARY = (400, 100, 600, 900)
# Upright person centred in PRIMARY (normalized coordinates)
BODY = {
    "nose": (0.50, 0.15, 0.0),
    "left_shoulder": (0.56, 0.25, 0.0), "right_shoulder": (0.44, 0.25, 0.0),
    "left_hip": (0.53, 0.52, 0.0), "right_hip": (0.47, 0.52, 0.0),
    "left_ankle": (0.52, 0.88, 0.0), "right_ankle": (0.48, 0.88, 0.0),
}


def landmarks(points=None, vis=None, shift_x=0.0, confidence=0.95, with_visibility=True) -> FrameLandmarks:
    kp = {k: (v[0] + shift_x, v[1], v[2]) for k, v in (points or BODY).items()}
    visibility = {k: 0.98 for k in kp}
    visibility.update(vis or {})
    return FrameLandmarks(
        height_ratio=0.8, shoulder_width=0.12, torso_length=0.27, hip_width=0.06, leg_length=0.36,
        yaw_angle=0.0, confidence=confidence, keypoints=kp, visibility=visibility if with_visibility else None,
    )


# The existing MediaPipe failure behaviour: contour fallback with no keypoints
CONTOUR_FALLBACK = FrameLandmarks(
    height_ratio=0.82, shoulder_width=0.28, torso_length=0.32, hip_width=0.24, leg_length=0.45,
    yaw_angle=0.0, confidence=0.6, keypoints={},
)


def validate(boxes, primary, lm, **cfg):
    return BodyValidator(enabled=cfg.pop("enabled", True), **cfg).validate(boxes, primary, IMG, lm)


class BodyValidatorTest(unittest.TestCase):
    # CASE A
    def test_valid_yolo_and_mediapipe(self):
        r = validate([PRIMARY], PRIMARY, landmarks())
        self.assertTrue(r.passed)
        self.assertEqual(r.reason, REASON_OK)
        self.assertTrue(r.yolo_person_detected and r.mediapipe_landmarks_detected and r.bbox_consistency)
        self.assertIsNone(r.user_message)

    # CASE D
    def test_yolo_missing_mediapipe_present_is_rejected(self):
        r = validate([], None, landmarks())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, REASON_YOLO_MISSING)
        self.assertFalse(r.yolo_person_detected)
        self.assertTrue(r.mediapipe_landmarks_detected)
        self.assertEqual(r.user_message, NO_PERSON_MESSAGE)

    # CASE B
    def test_yolo_present_mediapipe_missing_keeps_existing_behaviour(self):
        for lm in (None, CONTOUR_FALLBACK):
            r = validate([PRIMARY], PRIMARY, lm)
            self.assertTrue(r.passed)
            self.assertEqual(r.reason, REASON_MEDIAPIPE_UNAVAILABLE)
            self.assertFalse(r.mediapipe_landmarks_detected)
            self.assertIsNone(r.bbox_consistency)

    # CASE C
    def test_both_missing_is_rejected(self):
        for lm in (None, CONTOUR_FALLBACK):
            r = validate([], None, lm)
            self.assertFalse(r.passed)
            self.assertEqual(r.reason, REASON_NO_PERSON)
            self.assertEqual(r.user_message, NO_PERSON_MESSAGE)

    # CASE E
    def test_multiple_people_mediapipe_on_primary_passes(self):
        bystander = (700, 300, 800, 850)
        r = validate([bystander, PRIMARY], PRIMARY, landmarks())
        self.assertTrue(r.passed)
        self.assertEqual(r.person_count, 2)

    def test_multiple_people_mediapipe_on_other_person_is_rejected(self):
        bystander = (650, 100, 850, 900)
        r = validate([PRIMARY, bystander], PRIMARY, landmarks(shift_x=0.25))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, REASON_OTHER_PERSON)
        self.assertEqual(r.user_message, MATCH_MESSAGE)

    def test_wide_primary_box_containing_another_persons_landmarks(self):
        """Real zidane.jpg geometry: the primary's box also contains the other man, whom MediaPipe tracked."""
        wide_primary = (115, 197, 1114, 712)
        other = (748, 42, 1143, 713)
        pts = {"nose": (993, 210), "left_shoulder": (1011, 304), "right_shoulder": (857, 295),
               "left_hip": (935, 513), "right_hip": (830, 523)}
        lm = landmarks({k: (x / 1280, y / 720, 0.0) for k, (x, y) in pts.items()})
        r = BodyValidator(enabled=True).validate([wide_primary, other], wide_primary, (1280, 720), lm)
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, REASON_OTHER_PERSON)

    def test_multiple_people_result_independent_of_box_order(self):
        boxes = [PRIMARY, (650, 100, 850, 900), (0, 300, 150, 900)]
        outcomes = set()
        for seed in range(6):
            shuffled = boxes[:]
            random.Random(seed).shuffle(shuffled)
            outcomes.add((validate(shuffled, PRIMARY, landmarks()).reason,
                          validate(shuffled, PRIMARY, landmarks(shift_x=0.25)).reason))
        self.assertEqual(outcomes, {(REASON_OK, REASON_OTHER_PERSON)})

    def test_primary_is_taken_from_yolo_not_reselected(self):
        """Validator never picks its own primary: a primary missing from the box list is still the reference."""
        r = validate([(650, 100, 850, 900)], PRIMARY, landmarks())
        self.assertTrue(r.passed)
        self.assertEqual(r.person_count, 2)

    # inconsistent boxes
    def test_landmarks_outside_single_box_are_inconsistent(self):
        r = validate([(0, 0, 200, 300)], (0, 0, 200, 300), landmarks())
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, REASON_BBOX_INCONSISTENT)
        self.assertFalse(r.bbox_consistency)

    def test_landmarks_off_centre_of_single_box_are_inconsistent(self):
        wide = (200, 100, 900, 900)  # landmarks sit at x~500, centre is 550: offset 0.07
        self.assertTrue(validate([wide], wide, landmarks()).passed)
        r = validate([wide], wide, landmarks(shift_x=-0.22))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, REASON_BBOX_INCONSISTENT)
        self.assertGreater(r.center_offset, 0.25)

    # visibility / cropping
    def test_low_landmark_visibility_is_rejected(self):
        r = validate([PRIMARY], PRIMARY, landmarks(vis={"left_hip": 0.2}))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, REASON_LOW_VISIBILITY)
        self.assertAlmostEqual(r.min_landmark_visibility, 0.2)
        self.assertEqual(r.user_message, VISIBILITY_MESSAGE)

    def test_missing_required_keypoint_counts_as_invisible(self):
        lm = landmarks()
        lm.keypoints["nose"] = None
        self.assertEqual(validate([PRIMARY], PRIMARY, lm).reason, REASON_LOW_VISIBILITY)

    def test_visibility_falls_back_to_torso_confidence(self):
        self.assertTrue(validate([PRIMARY], PRIMARY, landmarks(with_visibility=False, confidence=0.9)).passed)
        r = validate([PRIMARY], PRIMARY, landmarks(with_visibility=False, confidence=0.3))
        self.assertEqual(r.reason, REASON_LOW_VISIBILITY)

    def test_required_landmark_outside_frame_is_cropped(self):
        pts = dict(BODY)
        pts["left_hip"] = (0.53, 1.04, 0.0)
        r = validate([PRIMARY], PRIMARY, landmarks(pts))
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, REASON_BODY_CROPPED)

    def test_optional_landmarks_do_not_block(self):
        """Ankles are not required by default: a knee-up capture with invisible feet still passes."""
        pts = dict(BODY)
        pts["left_ankle"] = (0.52, 1.3, 0.0)
        self.assertTrue(validate([PRIMARY], PRIMARY, landmarks(pts, vis={"left_ankle": 0.05})).passed)

    # configuration
    def test_bbox_tolerance_configuration(self):
        tight = (430, 120, 570, 500)  # hips at y=520 sit just below the box
        self.assertEqual(validate([tight], tight, landmarks(), bbox_tolerance=0.0).reason, REASON_BBOX_INCONSISTENT)
        self.assertTrue(validate([tight], tight, landmarks(), bbox_tolerance=0.20).passed)

    def test_center_tolerance_configuration(self):
        lm = landmarks(shift_x=-0.05)  # offset 0.25 of a 200px box
        self.assertTrue(validate([PRIMARY], PRIMARY, lm, center_tolerance=0.30).passed)
        self.assertEqual(validate([PRIMARY], PRIMARY, lm, center_tolerance=0.10).reason, REASON_BBOX_INCONSISTENT)

    def test_min_visibility_configuration(self):
        lm = landmarks(vis={"nose": 0.6})
        self.assertTrue(validate([PRIMARY], PRIMARY, lm, min_visibility=0.5).passed)
        self.assertEqual(validate([PRIMARY], PRIMARY, lm, min_visibility=0.7).reason, REASON_LOW_VISIBILITY)

    def test_required_landmarks_configuration(self):
        waist_up = landmarks(vis={"left_hip": 0.1, "right_hip": 0.1})
        self.assertEqual(validate([PRIMARY], PRIMARY, waist_up).reason, REASON_LOW_VISIBILITY)
        self.assertTrue(validate([PRIMARY], PRIMARY, waist_up,
                                 required_landmarks=["nose", "left_shoulder", "right_shoulder"]).passed)

    def test_required_landmarks_parsing(self):
        self.assertEqual(parse_required_landmarks(" Nose , left_shoulder,nose"), ("nose", "left_shoulder"))
        self.assertEqual(parse_required_landmarks("nose,elbow"), ("nose",))
        self.assertEqual(parse_required_landmarks(""), bv_module.DEFAULT_REQUIRED_LANDMARKS)
        self.assertEqual(parse_required_landmarks("chin"), bv_module.DEFAULT_REQUIRED_LANDMARKS)

    def test_defaults_come_from_config(self):
        with patch.object(bv_module, "BODY_BBOX_TOLERANCE", 0.33), \
             patch.object(bv_module, "BODY_CENTER_TOLERANCE", 0.44), \
             patch.object(bv_module, "BODY_MIN_LANDMARK_VISIBILITY", 0.66), \
             patch.object(bv_module, "BODY_REQUIRED_LANDMARKS", "nose,left_hip"), \
             patch.object(bv_module, "BODY_VALIDATION_ENABLED", False):
            v = BodyValidator()
        self.assertEqual((v.bbox_tolerance, v.center_tolerance, v.min_visibility), (0.33, 0.44, 0.66))
        self.assertEqual(v.required_landmarks, ("nose", "left_hip"))
        self.assertFalse(v.enabled)

    # disabled / unavailable
    def test_validation_disabled(self):
        for boxes, primary, lm in (([], None, None), ([PRIMARY], PRIMARY, landmarks(shift_x=0.4))):
            r = validate(boxes, primary, lm, enabled=False)
            self.assertTrue(r.passed)
            self.assertEqual(r.reason, REASON_DISABLED)

    def test_no_yolo_result_fails_open(self):
        r = validate(None, None, landmarks())
        self.assertTrue(r.passed)
        self.assertEqual(r.reason, REASON_YOLO_UNAVAILABLE)
        self.assertIsNone(r.yolo_person_detected)

    # logging
    def test_structured_log_has_flags_and_no_coordinates(self):
        line = validate([PRIMARY], PRIMARY, landmarks()).log_line()
        for field in ("yolo_person_detected=true", "mediapipe_landmarks_detected=true",
                      "bbox_consistency=true", "validation_passed=true", "reason=ok"):
            self.assertIn(field, line)
        self.assertNotIn("400", line)
        self.assertNotIn("0.15", line)
        self.assertIn("mediapipe_landmarks_detected=false", validate([PRIMARY], PRIMARY, None).log_line())


# ── router integration ──────────────────────────────────────────────────────
W, H = 384, 512


def _studio() -> np.ndarray:
    img = np.full((H, W, 3), 238, dtype=np.uint8)
    cv2.rectangle(img, (int(W * 0.35), int(H * 0.15)), (int(W * 0.65), int(H * 0.95)), (90, 60, 40), -1)
    return img


def _jpeg(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _uri(arr) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(arr)).decode()


CAPTURE = _studio()
GEMINI_OUTPUT = _jpeg(_studio())
CLOTH = ClothAnalysis(
    category="shirt", garment_name="Test Shirt", fabric_type="cotton", structure_level="soft_drape",
    is_outerwear=False, sleeve_behavior="natural", tryon_guidelines="fit naturally",
)
PASSING = SilhouetteVerificationResult(score=0.9, passed=True, threshold=0.6, details={})
MEDIAPIPE_FACE_MASK = np.zeros((H, W), np.float32)
ROUTER_PRIMARY = (134, 60, 250, 490)  # the dark body in CAPTURE
# Landmarks on that body, normalized to the capture
ROUTER_BODY = {k: (0.5 + (v[0] - 0.5) * 0.5, 0.15 + v[1] * 0.8, 0.0) for k, v in BODY.items()}
COMPOSITING_CALLS = [
    (Image, "blend"), (Image, "alpha_composite"), (Image, "composite"), (Image.Image, "paste"),
    (cv2, "addWeighted"), (cv2, "seamlessClone"), (cv2, "warpAffine"),
]


def yolo(boxes, primary=None):
    dets = [PersonDetection(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=0.9, area=(b[2] - b[0]) * (b[3] - b[1]))
            for b in boxes]
    svc = MagicMock()
    svc.detect_person.return_value = YOLODetectionResult(
        detections=dets, primary_person=YOLOService.select_primary_person(dets),
        count=len(dets), image_width=W, image_height=H,
    )
    return svc


class TryOnBodyValidationTest(unittest.TestCase):
    def _run(self, yolo_service, validation_lm, enabled=True, yolo_enabled=True, lm_error=None):
        """Runs /api/tryon. The first extract_single_frame call (validation) returns validation_lm; later
        pipeline calls (verifier, cross-fit) return None as in the other router tests."""
        capture_cache.clear()  # the same capture is run under different settings within a test
        calls, lm_calls = [], []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return GEMINI_OUTPUT, 0.97, CLOTH

        def fake_extract(img):
            lm_calls.append(img)
            if lm_error and len(lm_calls) == 1:
                raise lm_error
            return validation_lm if len(lm_calls) == 1 and enabled and yolo_enabled else None

        req = TryOnRequest(user_image=_uri(CAPTURE), dress_image=_uri(_studio()), cloth_analysis=CLOTH)
        spies = {}
        with ExitStack() as stack:
            for target, attr, kw in (
                (tryon, "YOLO_VALIDATION_ENABLED", {"new": yolo_enabled}),
                (tryon, "_get_yolo_service", {"return_value": yolo_service}),
                (tryon, "body_validator", {"new": BodyValidator(enabled=enabled)}),
                (tryon, "generate_tryon_image", {"side_effect": fake_generate}),
                (tryon, "analyze_user_appearance", {"return_value": ""}),
                (tryon, "_capture_has_face", {"return_value": True}),
                (tryon, "prepare_user_portrait", {"side_effect": lambda im: im}),
                (tryon.silhouette_verifier, "verify", {"return_value": PASSING}),
                (tryon.body_profile_service, "extract_single_frame", {"side_effect": fake_extract}),
                (tryon.body_profile_service, "extract_profile", {"side_effect": RuntimeError("n/a")}),
                (tryon.body_profile_service, "generate_face_and_hands_mask", {"return_value": (None, MEDIAPIPE_FACE_MASK)}),
                (tryon, "_face_score", {"return_value": 0.99}),
                (tryon, "apply_watermark", {}),
                (tryon, "generate_qr_data_uri", {"return_value": "qr"}),
            ):
                stack.enter_context(patch.object(target, attr, **kw))
            for owner, name in COMPOSITING_CALLS:
                original = getattr(owner, name)
                kw = {"wraps": original} if owner is cv2 else {"autospec": True, "side_effect": original}
                spies[f"{owner.__name__}.{name}"] = stack.enter_context(patch.object(owner, name, **kw))
            try:
                resp = asyncio.run(tryon.virtual_tryon(req))
            except HTTPException as exc:
                resp = exc
        return resp, calls, lm_calls, {k: s.call_count for k, s in spies.items()}

    def _assert_rejected(self, resp, calls, message):
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.detail, message)
        self.assertEqual(calls, [], "Gemini must not be called for a rejected capture")

    def test_case_a_valid_continues(self):
        resp, calls, lm_calls, _ = self._run(yolo([ROUTER_PRIMARY]), landmarks(ROUTER_BODY))
        self.assertTrue(resp.success)
        self.assertEqual(len(calls), 1)
        # validation's single MediaPipe pass ran on the full, unmodified capture
        self.assertTrue(np.array_equal(np.array(lm_calls[0]), np.array(tryon.clean_and_decode_base64(_uri(CAPTURE)))))

    def test_case_b_mediapipe_failure_uses_existing_behaviour(self):
        for lm, err in ((CONTOUR_FALLBACK, None), (None, RuntimeError("mediapipe crashed"))):
            resp, calls, _, _ = self._run(yolo([ROUTER_PRIMARY]), lm, lm_error=err)
            self.assertTrue(resp.success)
            self.assertEqual(len(calls), 1)

    def test_case_c_no_person_rejected_before_mediapipe_and_gemini(self):
        resp, calls, lm_calls, _ = self._run(yolo([]), landmarks(ROUTER_BODY))
        self._assert_rejected(resp, calls, NO_PERSON_MESSAGE)
        self.assertEqual(lm_calls, [])

    def test_case_e_other_person_tracked_is_rejected(self):
        # Primary on the left; MediaPipe landmarks sit on the smaller person on the right
        left, right = (20, 40, 200, 500), (250, 120, 370, 480)
        tracked_right = landmarks({k: (v[0] * 0.3 + 0.65 - 0.15, v[1], 0.0) for k, v in BODY.items()})
        resp, calls, _, _ = self._run(yolo([right, left]), tracked_right)
        self._assert_rejected(resp, calls, MATCH_MESSAGE)

    def test_low_visibility_rejected(self):
        resp, calls, _, _ = self._run(yolo([ROUTER_PRIMARY]), landmarks(ROUTER_BODY, vis={"right_hip": 0.1}))
        self._assert_rejected(resp, calls, VISIBILITY_MESSAGE)

    def test_disabled_adds_no_mediapipe_pass(self):
        _, _, lm_on, _ = self._run(yolo([ROUTER_PRIMARY]), landmarks(ROUTER_BODY), enabled=True)
        resp, calls, lm_off, _ = self._run(yolo([ROUTER_PRIMARY]), landmarks(ROUTER_BODY, vis={"nose": 0.0}), enabled=False)
        self.assertTrue(resp.success)
        self.assertEqual(len(lm_on) - len(lm_off), 1, "validation must add exactly one MediaPipe pass")

    def test_yolo_disabled_skips_body_validation(self):
        resp, calls, lm_calls, _ = self._run(MagicMock(side_effect=AssertionError), None, yolo_enabled=False)
        self.assertTrue(resp.success)
        _, _, lm_with_validation, _ = self._run(yolo([ROUTER_PRIMARY]), landmarks(ROUTER_BODY))
        self.assertEqual(len(lm_with_validation) - len(lm_calls), 1)

    def test_yolo_runs_once_per_request(self):
        svc = yolo([ROUTER_PRIMARY])
        self._run(svc, landmarks(ROUTER_BODY))
        svc.detect_person.assert_called_once()

    # ── FACE REGRESSION ──────────────────────────────────────────────────
    def test_face_regression_validation_changes_nothing_visual(self):
        on, calls_on, _, spies_on = self._run(yolo([ROUTER_PRIMARY]), landmarks(ROUTER_BODY), enabled=True)
        off, calls_off, _, spies_off = self._run(yolo([ROUTER_PRIMARY]), landmarks(ROUTER_BODY), enabled=False)
        served_on = base64.b64decode(on.result_image.split(",", 1)[1])
        served_off = base64.b64decode(off.result_image.split(",", 1)[1])
        self.assertEqual(served_on, GEMINI_OUTPUT)
        self.assertEqual(served_on, served_off)
        self.assertEqual(spies_on, spies_off)
        self.assertEqual(sum(spies_on.values()), 0, "validation must not composite anything")
        self.assertIs(calls_on[0]["feathered_mask"], MEDIAPIPE_FACE_MASK)
        self.assertTrue(np.array_equal(np.array(calls_on[0]["user_img"]), np.array(calls_off[0]["user_img"])))

    def test_face_regression_static_source_guard(self):
        app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
        pattern = re.compile(r"\b(blend|alpha_composite|composite|addWeighted|seamlessClone|warpAffine|fillPoly|rectangle|circle)\s*\(|\.paste\s*\(")
        with open(os.path.join(app_dir, "services", "body_validation.py"), encoding="utf-8") as fh:
            self.assertIsNone(pattern.search(fh.read()))
        for rel in ("face_preservation.py", "gemini_client.py"):
            with open(os.path.join(app_dir, "services", rel), encoding="utf-8") as fh:
                self.assertNotIn("body_validation", fh.read())


# ── real models (no Gemini) ────────────────────────────────────────────────
class RealModelBodyValidationTest(unittest.TestCase):
    """Real YOLOv8n + MediaPipe Pose on Ultralytics' bundled sample photos."""

    @classmethod
    def setUpClass(cls):
        ASSETS = os.path.join(os.path.dirname(__file__), "assets")
        cls.bus = Image.open(os.path.join(ASSETS, "bus.jpg")).convert("RGB")
        cls.zidane = Image.open(os.path.join(ASSETS, "zidane.jpg")).convert("RGB")

    def _check(self, img):
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", True), \
             patch.object(tryon, "body_validator", BodyValidator(enabled=True)):
            return tryon._validate_body(img, tryon._validate_person(img))

    def test_single_full_body_person_passes(self):
        before = np.array(self.bus).copy()
        result = self._check(self.bus.crop((20, 200, 320, 920)))
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, REASON_OK)
        self.assertTrue(np.array_equal(np.array(self.bus), before))

    def test_multi_person_scene_with_mediapipe_on_primary_passes(self):
        self.assertEqual(self._check(self.bus).reason, REASON_OK)

    def test_mediapipe_tracking_non_primary_person_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            self._check(self.zidane)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail, MATCH_MESSAGE)


if __name__ == "__main__":
    unittest.main()
