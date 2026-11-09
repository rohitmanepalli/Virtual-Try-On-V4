"""
YOLO + MediaPipe body cross-validation.

YOLO answers "is there a person, and which one is the primary?"; MediaPipe answers "where are
the body landmarks?". Before a try-on is generated, this layer checks that the two agree on the
same (primary) person and that the body is visible enough to dress.

Pure logic on already-computed results: it runs no models and never touches image pixels.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import (
    BODY_BBOX_TOLERANCE,
    BODY_CENTER_TOLERANCE,
    BODY_MIN_LANDMARK_VISIBILITY,
    BODY_REQUIRED_LANDMARKS,
    BODY_VALIDATION_ENABLED,
)
from app.services.body_profile import FrameLandmarks

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels

# Landmarks for which FrameLandmarks carries both a keypoint and a visibility score
KNOWN_LANDMARKS = (
    "nose", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_ankle", "right_ankle",
)
DEFAULT_REQUIRED_LANDMARKS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")

REASON_OK = "ok"
REASON_DISABLED = "disabled"
REASON_YOLO_UNAVAILABLE = "yolo_unavailable"
REASON_MEDIAPIPE_UNAVAILABLE = "mediapipe_unavailable"
REASON_NO_PERSON = "no_person"
REASON_YOLO_MISSING = "yolo_missing_mediapipe_present"
REASON_LOW_VISIBILITY = "low_landmark_visibility"
REASON_BODY_CROPPED = "body_cropped"
REASON_OTHER_PERSON = "landmarks_match_other_person"
REASON_BBOX_INCONSISTENT = "bbox_inconsistent"

NO_PERSON_MESSAGE = "We couldn't detect a person in your photo. Please stand fully in frame facing the camera and try again."
VISIBILITY_MESSAGE = "We couldn't see enough of your body. Please make sure your head, shoulders and hips are clearly visible in the frame, then try again."
MATCH_MESSAGE = "We couldn't clearly identify you in the photo. Please make sure only you are in the frame and try again."

USER_MESSAGES = {
    REASON_NO_PERSON: NO_PERSON_MESSAGE,
    REASON_YOLO_MISSING: NO_PERSON_MESSAGE,
    REASON_LOW_VISIBILITY: VISIBILITY_MESSAGE,
    REASON_BODY_CROPPED: VISIBILITY_MESSAGE,
    REASON_OTHER_PERSON: MATCH_MESSAGE,
    REASON_BBOX_INCONSISTENT: MATCH_MESSAGE,
}


def parse_required_landmarks(raw: Optional[str]) -> Tuple[str, ...]:
    """Comma-separated landmark names -> tuple of known names (unknown names dropped with a warning)."""
    names = [n.strip().lower() for n in (raw or "").split(",") if n.strip()]
    unknown = [n for n in names if n not in KNOWN_LANDMARKS]
    if unknown:
        print(f"[BODY VALIDATION] Ignoring unknown required landmarks {unknown}; known: {list(KNOWN_LANDMARKS)}")
    known = tuple(dict.fromkeys(n for n in names if n in KNOWN_LANDMARKS))
    return known or DEFAULT_REQUIRED_LANDMARKS


@dataclass
class BodyValidationResult:
    """Internal validation outcome. Contains no image data, coordinates or face information."""

    passed: bool
    reason: str
    yolo_person_detected: Optional[bool]  # None: no YOLO result to compare against
    mediapipe_landmarks_detected: bool
    bbox_consistency: Optional[bool] = None  # None: not evaluated
    person_count: int = 0
    min_landmark_visibility: Optional[float] = None
    center_offset: Optional[float] = None

    @property
    def user_message(self) -> Optional[str]:
        return None if self.passed else USER_MESSAGES.get(self.reason, MATCH_MESSAGE)

    def log_line(self) -> str:
        def fmt(v):
            return "n/a" if v is None else str(v).lower()
        return (
            "[BODY VALIDATION] body_validation:"
            f" yolo_person_detected={fmt(self.yolo_person_detected)}"
            f" mediapipe_landmarks_detected={fmt(self.mediapipe_landmarks_detected)}"
            f" bbox_consistency={fmt(self.bbox_consistency)}"
            f" validation_passed={fmt(self.passed)}"
            f" reason={self.reason}"
            f" person_count={self.person_count}"
            f" min_landmark_visibility={fmt(None if self.min_landmark_visibility is None else round(self.min_landmark_visibility, 2))}"
            f" center_offset={fmt(None if self.center_offset is None else round(self.center_offset, 3))}"
        )


class BodyValidator:
    """Cross-checks YOLO person detections against MediaPipe Pose landmarks (see module docstring)."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        bbox_tolerance: Optional[float] = None,
        center_tolerance: Optional[float] = None,
        min_visibility: Optional[float] = None,
        required_landmarks: Optional[Sequence[str]] = None,
    ):
        self.enabled = BODY_VALIDATION_ENABLED if enabled is None else bool(enabled)
        self.bbox_tolerance = max(0.0, float(BODY_BBOX_TOLERANCE if bbox_tolerance is None else bbox_tolerance))
        self.center_tolerance = max(0.0, float(BODY_CENTER_TOLERANCE if center_tolerance is None else center_tolerance))
        self.min_visibility = float(BODY_MIN_LANDMARK_VISIBILITY if min_visibility is None else min_visibility)
        self.required_landmarks = parse_required_landmarks(
            BODY_REQUIRED_LANDMARKS if required_landmarks is None else ",".join(required_landmarks)
        )

    def _visibility(self, landmarks: FrameLandmarks, name: str) -> float:
        if landmarks.keypoints.get(name) is None:
            return 0.0
        vis = landmarks.visibility.get(name) if landmarks.visibility else None
        # Without per-landmark scores, fall back to MediaPipe's torso confidence (mean shoulder/hip visibility)
        return float(landmarks.confidence if vis is None else vis)

    def _contains(self, box: Box, points: List[Tuple[float, float]]) -> bool:
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * self.bbox_tolerance, (y2 - y1) * self.bbox_tolerance
        return all(x1 - mx <= px <= x2 + mx and y1 - my <= py <= y2 + my for px, py in points)

    @staticmethod
    def _center_offset(box: Box, cx: float) -> float:
        x1, _, x2, _ = box
        return abs(cx - (x1 + x2) / 2.0) / float(max(x2 - x1, 1))

    def validate(
        self,
        person_boxes: Optional[Sequence[Box]],
        primary_box: Optional[Box],
        image_size: Tuple[int, int],
        landmarks: Optional[FrameLandmarks],
    ) -> BodyValidationResult:
        """
        person_boxes: every YOLO person box (None when no YOLO result exists, [] when YOLO found nobody).
        primary_box: YOLO's primary person (YOLOService.select_primary_person); never re-selected here.
        image_size: (width, height) of the image both models ran on.
        landmarks: MediaPipe result for that image; None or empty keypoints means MediaPipe failed.
        """
        mp_ok = bool(landmarks is not None and landmarks.keypoints)
        boxes = [tuple(int(v) for v in b) for b in (person_boxes or [])]
        primary = tuple(int(v) for v in primary_box) if primary_box is not None else None
        if primary is not None and primary not in boxes:
            boxes.append(primary)

        def result(passed, reason, **kw):
            return BodyValidationResult(
                passed=passed, reason=reason, yolo_person_detected=kw.pop("yolo", primary is not None),
                mediapipe_landmarks_detected=mp_ok, person_count=len(boxes), **kw,
            )

        if not self.enabled:
            return result(True, REASON_DISABLED, yolo=None if person_boxes is None else primary is not None)
        if person_boxes is None:
            # Nothing to cross-check (YOLO disabled or errored): keep the existing fail-open behaviour
            return result(True, REASON_YOLO_UNAVAILABLE, yolo=None)
        if primary is None:
            # Case C (nobody) / Case D (MediaPipe sees a body YOLO cannot confirm): unreliable either way
            return result(False, REASON_YOLO_MISSING if mp_ok else REASON_NO_PERSON)
        if not mp_ok:
            # Case B: existing MediaPipe failure behaviour (contour fallback) continues; no new measurement
            return result(True, REASON_MEDIAPIPE_UNAVAILABLE)

        visibilities = {name: self._visibility(landmarks, name) for name in self.required_landmarks}
        min_vis = min(visibilities.values())
        if min_vis < self.min_visibility:
            return result(False, REASON_LOW_VISIBILITY, min_landmark_visibility=min_vis)

        norm = [landmarks.keypoints[name] for name in self.required_landmarks]
        if any(not (0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0) for p in norm):
            return result(False, REASON_BODY_CROPPED, min_landmark_visibility=min_vis)

        img_w, img_h = image_size
        points = [(p[0] * img_w, p[1] * img_h) for p in norm]
        cx = sum(px for px, _ in points) / len(points)

        # Which YOLO person do these landmarks belong to? Containing box with the closest centre (primary wins ties).
        candidates = sorted(
            (self._center_offset(b, cx), b != primary, b) for b in boxes if self._contains(b, points)
        )
        primary_offset = self._center_offset(primary, cx)
        if not candidates:
            return result(False, REASON_BBOX_INCONSISTENT, bbox_consistency=False,
                          min_landmark_visibility=min_vis, center_offset=primary_offset)
        best_box = candidates[0][2]
        if best_box != primary:
            # Case E: MediaPipe locked onto someone other than YOLO's primary person
            return result(False, REASON_OTHER_PERSON, bbox_consistency=False,
                          min_landmark_visibility=min_vis, center_offset=primary_offset)
        if primary_offset > self.center_tolerance:
            return result(False, REASON_BBOX_INCONSISTENT, bbox_consistency=False,
                          min_landmark_visibility=min_vis, center_offset=primary_offset)
        return result(True, REASON_OK, bbox_consistency=True,
                      min_landmark_visibility=min_vis, center_offset=primary_offset)


# Global validator configured from environment
body_validator = BodyValidator()
