import math
import os
import threading
import urllib.request
import traceback
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from app.config import (
    CALIBRATED_SCALE_FACTOR_5_6M,
    CAPTURE_DISTANCE_MAX_METERS,
    CAPTURE_DISTANCE_MIN_METERS,
    CAPTURE_DISTANCE_OPTIMAL_METERS,
    MEASUREMENT_CALIBRATION_DISTANCE_M,
    MEASUREMENT_FOCAL_LENGTH_PX,
    MEASUREMENT_REFERENCE_HEIGHT_PX,
)
from app.models import BodyMeasurement, BodyMeasurements, BodyProfile
from app.services import request_timing

# Population-average torso sagittal depth / bi-acromial shoulder width. Body depth is not observable
# from a front-on capture; the chest/waist reference denominators and size bands are calibrated to this ratio.
SAGITTAL_DEPTH_TO_SHOULDER_RATIO = 0.72

# Informational cm display only (measure_body) — none of these feed sizing.
# Population-average shoulder landmark width / stature (biacromial ≈ 23% of height); height fallback when feet are out of frame.
SHOULDER_WIDTH_TO_HEIGHT_RATIO = 0.23
ANKLE_VISIBILITY_MIN = 0.6
# Chest and natural-waist lines as fractions of the shoulder→hip landmark span (from average body proportions)
CHEST_LINE_FRACTION = 0.33
WAIST_LINE_FRACTION = 0.70
# Silhouette width above this multiple of the shoulder-landmark width means the arms are merged into the torso
ARM_MERGE_MAX_WIDTH_RATIO = 1.15

# Torso fullness (torso_fullness): mean midsection silhouette width with the arms excluded, over the shoulder-joint
# width. Feeds the generation prompt and silhouette verifier only — size matching stays on the joint proportions.
# Rows sampled down the shoulder→hip landmark span, from the stomach to the hips. The mean over the band (not the
# widest row) is used because the widest row is usually the hips, which barely move when a render slims the stomach.
MIDSECTION_LINE_FRACTIONS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
# A mirrored arm half-width beyond this multiple of the shoulder width is not an arm (hand, bag, background merge)
ARM_HALF_WIDTH_MAX_RATIO = 0.35
TORSO_MIN_WIDTH_RATIO = 0.30
# Provisional (upper bound, shape) bands, set on 10 studio renders (leanest 0.89, stockiest 1.04); recalibrate on
# raw kiosk captures. Repeatability is ~1-4% per image under framing / JPEG / lighting changes.
TORSO_SHAPE_BANDS = ((0.92, "lean"), (1.02, "average"), (1.15, "full"), (math.inf, "very_full"))


def classify_torso_shape(torso_fullness: Optional[float]) -> Optional[str]:
    if torso_fullness is None:
        return None
    return next(shape for upper, shape in TORSO_SHAPE_BANDS if torso_fullness < upper)

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    _MEDIAPIPE_AVAILABLE = False


def _in_frame_visibility(p: Any) -> float:
    """MediaPipe visibility, or 0 when the normalized landmark lies outside the image."""
    if not (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0):
        return 0.0
    return float(getattr(p, 'visibility', 0.0) or 0.0)


class FrameLandmarks:
    """Stores key normalized measurements for a single frame."""
    def __init__(
        self,
        height_ratio: float,
        shoulder_width: float,
        torso_length: float,
        hip_width: float,
        leg_length: float,
        yaw_angle: float,
        confidence: float,
        keypoints: Dict[str, Tuple[float, float, float]],
        segmentation_mask: Optional[np.ndarray] = None,
        visibility: Optional[Dict[str, float]] = None,
        image_size: Optional[Tuple[int, int]] = None
    ):
        self.height_ratio = height_ratio
        self.shoulder_width = shoulder_width
        self.torso_length = torso_length
        self.hip_width = hip_width
        self.leg_length = leg_length
        self.yaw_angle = yaw_angle  # in degrees (0 = front, 90 = side)
        self.confidence = confidence
        self.keypoints = keypoints
        self.segmentation_mask = segmentation_mask
        # Per-landmark MediaPipe visibility (0-1); empty for the contour fallback. Used for body validation only.
        self.visibility = visibility or {}
        # (width, height) of the original capture; keypoints are normalized, so pixel lengths use this size
        self.image_size = image_size


class BodyProfileService:
    """
    Front-capture pose estimation and anthropometric measurement engine.
    Uses MediaPipe PoseLandmarker (33 3D landmarks + segmentation mask)
    with adaptive background contour fallback.
    """

    def __init__(self):
        self._detector = None
        self._face_detector = None
        # Try-ons run on worker threads, and one detector must not run two images at once
        self._pose_lock = threading.Lock()
        self.face_lock = threading.Lock()
        if _MEDIAPIPE_AVAILABLE:
            self._init_detector()

    def _init_detector(self):
        try:
            model_candidates = [
                os.path.join(os.path.dirname(__file__), "..", "..", "assets", "pose_landmarker_lite.task"),
                os.path.join(os.path.dirname(__file__), "..", "..", "pose_landmarker_lite.task"),
                "assets/pose_landmarker_lite.task",
                "pose_landmarker_lite.task",
            ]
            model_path = None
            for p in model_candidates:
                if os.path.exists(p):
                    model_path = os.path.abspath(p)
                    break

            if not model_path:
                model_path = os.path.abspath("assets/pose_landmarker_lite.task")
                os.makedirs("assets", exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=True,
                min_pose_detection_confidence=0.4
            )
            self._detector = vision.PoseLandmarker.create_from_options(options)
            print(f"[BODY PROFILE] MediaPipe PoseLandmarker initialized from: {model_path}")

            # Initialize Face Landmarker (Face Mesh 478 points) if available
            face_model_candidates = [
                os.path.join(os.path.dirname(__file__), "..", "..", "assets", "face_landmarker.task"),
                os.path.join(os.path.dirname(__file__), "..", "..", "face_landmarker.task"),
                "assets/face_landmarker.task",
                "face_landmarker.task",
            ]
            face_path = None
            for fp in face_model_candidates:
                if os.path.exists(fp):
                    face_path = os.path.abspath(fp)
                    break
            if face_path:
                f_opts = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=face_path),
                    output_face_blendshapes=False,
                    min_face_detection_confidence=0.4
                )
                self._face_detector = vision.FaceLandmarker.create_from_options(f_opts)
                print(f"[BODY PROFILE] MediaPipe FaceLandmarker initialized from: {face_path}")
        except Exception as e:
            print(f"[BODY PROFILE] MediaPipe PoseLandmarker initialization warning: {e}")
            traceback.print_exc()

    def extract_single_frame(self, image: Image.Image) -> Optional[FrameLandmarks]:
        """Extracts normalized keypoints and geometric ratios from a single image. Each image object runs the pose model
        once (the pipeline checks the same capture and render repeatedly and never changes their pixels); callers must
        treat the result as read-only."""
        cached = getattr(image, "_vto_landmarks", None)
        if cached is not None and cached[0] == image.size:
            return cached[1]
        with request_timing.step("mediapipe"):
            landmarks = self._extract_single_frame(image)
        image._vto_landmarks = (image.size, landmarks)
        return landmarks

    def _extract_single_frame(self, image: Image.Image) -> Optional[FrameLandmarks]:
        target_max = 1024
        scale = min(1.0, target_max / float(max(image.width, image.height, 1)))
        nw = max(64, (int(image.width * scale) // 4) * 4)
        nh = max(64, (int(image.height * scale) // 4) * 4)
        if (nw, nh) != (image.width, image.height):
            proc_img = image.resize((nw, nh), Image.Resampling.BILINEAR)
        else:
            proc_img = image
        img_np = np.ascontiguousarray(np.array(proc_img.convert("RGB")), dtype=np.uint8)
        h, w, _ = img_np.shape

        if self._detector:
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                with self._pose_lock:
                    results = self._detector.detect(mp_img)
                if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
                    lm = results.pose_landmarks[0]
                    nose = (lm[0].x, lm[0].y, lm[0].z)
                    l_sh = (lm[11].x, lm[11].y, lm[11].z)
                    r_sh = (lm[12].x, lm[12].y, lm[12].z)

                    def _get_kp(idx: int):
                        if len(lm) > idx:
                            p = lm[idx]
                            vis = getattr(p, 'visibility', 0.0) or 0.0
                            pres = getattr(p, 'presence', 0.0) or 0.0
                            if vis >= 0.75 and pres >= 0.70:
                                return (p.x, p.y, p.z)
                        return None

                    # Elbows are kept unconditionally (visibility recorded below): arms hanging at the sides merge
                    # into the torso silhouette, and the arm centre line is what separates them (torso_widths)
                    l_elbow = (lm[13].x, lm[13].y, lm[13].z)
                    r_elbow = (lm[14].x, lm[14].y, lm[14].z)
                    l_wrist = _get_kp(15)
                    r_wrist = _get_kp(16)
                    l_pinky = _get_kp(17)
                    r_pinky = _get_kp(18)
                    l_index = _get_kp(19)
                    r_index = _get_kp(20)
                    l_hip = (lm[23].x, lm[23].y, lm[23].z)
                    r_hip = (lm[24].x, lm[24].y, lm[24].z)
                    l_ank = (lm[27].x, lm[27].y, lm[27].z)
                    r_ank = (lm[28].x, lm[28].y, lm[28].z)

                    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2, (l_sh[2] + r_sh[2]) / 2)
                    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2, (l_hip[2] + r_hip[2]) / 2)
                    mid_ank = ((l_ank[0] + r_ank[0]) / 2, (l_ank[1] + r_ank[1]) / 2, (l_ank[2] + r_ank[2]) / 2)

                    # Estimate head top
                    head_top_y = max(0.0, nose[1] - abs(mid_sh[1] - nose[1]) * 1.1)
                    height_ratio = abs(mid_ank[1] - head_top_y)

                    shoulder_width = math.sqrt((r_sh[0] - l_sh[0]) ** 2 + (r_sh[1] - l_sh[1]) ** 2)
                    hip_width = math.sqrt((r_hip[0] - l_hip[0]) ** 2 + (r_hip[1] - l_hip[1]) ** 2)
                    torso_length = math.sqrt((mid_hip[0] - mid_sh[0]) ** 2 + (mid_hip[1] - mid_sh[1]) ** 2)
                    leg_length = math.sqrt((mid_ank[0] - mid_hip[0]) ** 2 + (mid_ank[1] - mid_hip[1]) ** 2)

                    # Estimate yaw angle
                    z_diff = l_sh[2] - r_sh[2]
                    norm_z = max(-1.0, min(1.0, z_diff / max(shoulder_width, 0.01)))
                    yaw_angle = abs(math.degrees(math.asin(norm_z))) if abs(norm_z) <= 1.0 else 0.0

                    vis_11 = getattr(lm[11], 'visibility', 0.95)
                    vis_12 = getattr(lm[12], 'visibility', 0.95)
                    vis_23 = getattr(lm[23], 'visibility', 0.95)
                    vis_24 = getattr(lm[24], 'visibility', 0.95)
                    conf = (vis_11 + vis_12 + vis_23 + vis_24) / 4.0

                    seg_mask = None
                    if results.segmentation_masks and len(results.segmentation_masks) > 0:
                        try:
                            seg_mask = np.array(results.segmentation_masks[0].numpy_view(), copy=True, dtype=np.float32)
                        except Exception:
                            seg_mask = None

                    kp_dict = {
                        "nose": nose,
                        "left_shoulder": l_sh,
                        "right_shoulder": r_sh,
                        "left_elbow": l_elbow,
                        "right_elbow": r_elbow,
                        "left_wrist": l_wrist,
                        "right_wrist": r_wrist,
                        "left_pinky": l_pinky,
                        "right_pinky": r_pinky,
                        "left_index": l_index,
                        "right_index": r_index,
                        "left_hip": l_hip,
                        "right_hip": r_hip,
                        "left_ankle": l_ank,
                        "right_ankle": r_ank,
                    }

                    vis_indices = {
                        "nose": 0, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
                        "left_wrist": 15, "right_wrist": 16,
                        "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
                        "left_ankle": 27, "right_ankle": 28,
                    }
                    # MediaPipe extrapolates landmarks past the image edge (legs below a thigh-level crop) and can
                    # still give them a fair visibility score; a landmark outside the frame is not visible
                    visibility = {
                        name: _in_frame_visibility(lm[idx])
                        for name, idx in vis_indices.items() if len(lm) > idx
                    }

                    return FrameLandmarks(
                        height_ratio=height_ratio,
                        shoulder_width=shoulder_width,
                        torso_length=torso_length,
                        hip_width=hip_width,
                        leg_length=leg_length,
                        yaw_angle=yaw_angle,
                        confidence=conf,
                        keypoints=kp_dict,
                        segmentation_mask=seg_mask,
                        visibility=visibility,
                        image_size=(image.width, image.height)
                    )
            except Exception as e:
                print(f"[BODY PROFILE] MediaPipe PoseLandmarker execution warning: {e}")

        # Geometric contour-based fallback with adaptive background detection
        return self._geometric_contour_fallback(img_np)

    def _geometric_contour_fallback(self, img_np: np.ndarray) -> FrameLandmarks:
        """Adaptive fallback analyzing silhouette bounds against background corners."""
        h, w, _ = img_np.shape
        gray = np.mean(img_np, axis=2)

        # Check background corner intensity
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        bg_val = float(np.median(corners))
        if bg_val > 220:
            mask = gray < 235
        elif bg_val < 35:
            mask = gray > 45
        else:
            diff = np.abs(gray - bg_val)
            mask = diff > 25

        y_indices, x_indices = np.where(mask)
        if len(y_indices) < 100:
            return FrameLandmarks(
                height_ratio=0.82,
                shoulder_width=0.28,
                torso_length=0.32,
                hip_width=0.24,
                leg_length=0.45,
                yaw_angle=0.0,
                confidence=0.6,
                keypoints={}
            )

        top_y, bot_y = np.min(y_indices) / h, np.max(y_indices) / h
        height_ratio = max(0.4, min(0.95, bot_y - top_y))

        shoulder_y = int((top_y + height_ratio * 0.20) * h)
        hip_y = int((top_y + height_ratio * 0.55) * h)

        row_sh = np.where(mask[max(0, min(h-1, shoulder_y)), :])[0]
        row_hip = np.where(mask[max(0, min(h-1, hip_y)), :])[0]

        sh_w = (np.max(row_sh) - np.min(row_sh)) / w if len(row_sh) > 0 else 0.28
        sh_w = max(0.15, min(0.45, sh_w))  # clamp to realistic human bounds

        hip_w = (np.max(row_hip) - np.min(row_hip)) / w if len(row_hip) > 0 else 0.24
        hip_w = max(0.12, min(0.40, hip_w))

        torso_len = height_ratio * 0.35
        leg_len = height_ratio * 0.45

        return FrameLandmarks(
            height_ratio=height_ratio,
            shoulder_width=sh_w,
            torso_length=torso_len,
            hip_width=hip_w,
            leg_length=leg_len,
            yaw_angle=0.0,
            confidence=0.65,
            keypoints={}
        )

    def extract_profile(self, image: Image.Image, hold_frames: Optional[List[Image.Image]] = None) -> BodyProfile:
        """
        Extracts the body profile from a single front-on capture.
        hold_frames are extra frames grabbed within ~0.5s of the capture while the visitor holds still;
        each measurement is the median across all frames to suppress per-frame landmark jitter.
        Body depth is not observable from a front view, so circumferences use a population-average
        depth-to-width ratio (SAGITTAL_DEPTH_TO_SHOULDER_RATIO).
        """
        return self.profile_from_frames(self.extract_frames(image, hold_frames))

    def extract_frames(self, image: Image.Image, hold_frames: Optional[List[Image.Image]] = None) -> List[FrameLandmarks]:
        """Runs pose detection once per capture frame; the result feeds both profile_from_frames and measure_body."""
        frames = [image] + list(hold_frames or [])

        extracted: List[FrameLandmarks] = []
        for frame in frames:
            lm = self.extract_single_frame(frame)
            if lm and lm.confidence >= 0.4:
                extracted.append(lm)

        if not extracted:
            extracted.append(self._geometric_contour_fallback(np.array(image.convert("RGB"))))
        return extracted

    def profile_from_frames(self, extracted: List[FrameLandmarks]) -> BodyProfile:
        """Size-relevant BodyProfile (proportion ratios) from extracted capture frames."""
        # Width measurements only from frames where the visitor squarely faces the camera
        frontal_frames = [f for f in extracted if f.yaw_angle < 35.0] or extracted

        height = self._median([f.height_ratio for f in extracted])
        transverse_shoulder = self._median([f.shoulder_width for f in frontal_frames])
        avg_torso_length = self._median([f.torso_length for f in frontal_frames])
        transverse_hip = self._median([f.hip_width for f in frontal_frames])

        torso_aspect_ratio = avg_torso_length / max(transverse_shoulder, 0.05)
        sh_to_hip = transverse_shoulder / max(transverse_hip, 0.05)

        sagittal_depth = transverse_shoulder * SAGITTAL_DEPTH_TO_SHOULDER_RATIO

        # Elliptical circumferences
        chest_transverse = transverse_shoulder * 0.95
        chest_sagittal = sagittal_depth * 1.05
        chest_circ = math.pi * math.sqrt(2 * ((chest_transverse / 2) ** 2 + (chest_sagittal / 2) ** 2))

        waist_transverse = min(transverse_shoulder, transverse_hip) * 0.82
        waist_sagittal = sagittal_depth * 0.85
        waist_circ = math.pi * math.sqrt(2 * ((waist_transverse / 2) ** 2 + (waist_sagittal / 2) ** 2))

        hip_transverse = transverse_hip
        hip_sagittal = sagittal_depth * 1.0
        hip_circ = math.pi * math.sqrt(2 * ((hip_transverse / 2) ** 2 + (hip_sagittal / 2) ** 2))

        # Standard size-band reference circumferences (calibrated for kiosk distance where M: chest=1.25, waist=0.85)
        # Prevents self-canceling ratios (chest_circ / hip_circ) that previously flattened all body profiles to M.
        ref_chest_denom = 0.561
        ref_waist_denom = 0.645

        norm_chest = round(chest_circ / ref_chest_denom, 3)
        norm_waist = round(waist_circ / ref_waist_denom, 3)
        norm_hip = 1.0

        body_type = self._classify_body_type(sh_to_hip, norm_chest, norm_waist, norm_hip)
        avg_conf = float(np.mean([f.confidence for f in extracted]))

        fullness_values = [v for v in (self.torso_fullness(f) for f in frontal_frames) if v is not None]
        torso_fullness = round(self._median(fullness_values), 3) if fullness_values else None

        # Metric conversions calibrated for 5–6m kiosk capture distance
        est_height_cm = round(float(height * CALIBRATED_SCALE_FACTOR_5_6M), 1)
        est_shoulder_cm = round(float(transverse_shoulder * CALIBRATED_SCALE_FACTOR_5_6M), 1)
        est_chest_circ_cm = round(float(chest_circ * CALIBRATED_SCALE_FACTOR_5_6M), 1)
        est_waist_circ_cm = round(float(waist_circ * CALIBRATED_SCALE_FACTOR_5_6M), 1)
        est_hip_circ_cm = round(float(hip_circ * CALIBRATED_SCALE_FACTOR_5_6M), 1)

        return BodyProfile(
            height_ratio=round(float(height), 3),
            shoulder_width_ratio=round(float(transverse_shoulder), 3),
            torso_aspect_ratio=round(float(torso_aspect_ratio), 3),
            shoulder_to_hip_ratio=round(float(sh_to_hip), 3),
            chest_proportion=round(float(norm_chest), 3),
            waist_proportion=round(float(norm_waist), 3),
            hip_proportion=round(float(norm_hip), 3),
            body_type=body_type,
            confidence_score=round(avg_conf, 2),
            frame_count_sampled=len(extracted),
            torso_fullness=torso_fullness,
            torso_shape=classify_torso_shape(torso_fullness),
            keypoints_summary={
                "frontal_frames_count": len(frontal_frames),
                "torso_fullness_frames": len(fullness_values),
                "transverse_shoulder": round(float(transverse_shoulder), 3),
                "sagittal_depth_est": round(float(sagittal_depth), 3),
                "depth_to_width_ratio": SAGITTAL_DEPTH_TO_SHOULDER_RATIO,
                "capture_distance_range_meters": [CAPTURE_DISTANCE_MIN_METERS, CAPTURE_DISTANCE_MAX_METERS],
                "calibrated_scale_factor": CALIBRATED_SCALE_FACTOR_5_6M,
                "estimated_measurements_cm": {
                    "height": est_height_cm,
                    "shoulder_width": est_shoulder_cm,
                    "chest_circumference": est_chest_circ_cm,
                    "waist_circumference": est_waist_circ_cm,
                    "hip_circumference": est_hip_circ_cm,
                },
            }
        )

    def _median(self, values: List[float]) -> float:
        """Aggregates one measurement across the capture frames (median rejects a single jittery frame)."""
        return float(np.median(values))

    def measure_body(self, extracted: List[FrameLandmarks]) -> BodyMeasurements:
        """
        Informational cm estimates for display only — never used for sizing or try-on.
        Each value is the median across capture frames. Scale assumes the visitor stands at
        MEASUREMENT_CALIBRATION_DISTANCE_M; anywhere in the 2–3m zone leaves up to ±20% scale error,
        so every value is "estimated" (the visitor can correct height/shoulder, which rescales the rest).
        """
        per_frame = [m for m in (self._frame_measurements_cm(f) for f in extracted) if m]

        def pick(key: str) -> Optional[float]:
            vals = [m[key] for m in per_frame if m[key] is not None]
            return self._median(vals) if vals else None

        def build(value: Optional[float], source: str) -> Optional[BodyMeasurement]:
            return BodyMeasurement(value_cm=round(value, 1), source=source) if value is not None else None

        shoulder = pick("shoulder_width")
        height = build(pick("height"), "silhouette")
        if height is None and shoulder is not None:
            # Feet out of frame: fall back to the population-average shoulder-to-height ratio
            height = build(shoulder / SHOULDER_WIDTH_TO_HEIGHT_RATIO, "shoulder_ratio")

        return BodyMeasurements(
            height=height,
            shoulder_width=build(shoulder, "landmarks"),
            chest_width=build(pick("chest_width"), "silhouette"),
            waist_width=build(pick("waist_width"), "silhouette"),
            assumed_distance_m=MEASUREMENT_CALIBRATION_DISTANCE_M,
        )

    def _frame_measurements_cm(self, f: FrameLandmarks) -> Optional[Dict[str, Optional[float]]]:
        """Per-frame cm values (None where not measurable); None for frames without pose landmarks."""
        kp = f.keypoints
        l_sh, r_sh = kp.get("left_shoulder"), kp.get("right_shoulder")
        l_hip, r_hip = kp.get("left_hip"), kp.get("right_hip")
        if not f.image_size or not (l_sh and r_sh and l_hip and r_hip):
            return None
        img_w, img_h = f.image_size
        # Pinhole model: cm = px × distance / focal length (focal length scales with capture resolution)
        focal_px = MEASUREMENT_FOCAL_LENGTH_PX * img_h / MEASUREMENT_REFERENCE_HEIGHT_PX
        cm_per_px = MEASUREMENT_CALIBRATION_DISTANCE_M * 100.0 / focal_px

        shoulder_px = math.hypot((r_sh[0] - l_sh[0]) * img_w, (r_sh[1] - l_sh[1]) * img_h)
        result: Dict[str, Optional[float]] = {
            "shoulder_width": shoulder_px * cm_per_px, "height": None, "chest_width": None, "waist_width": None,
        }

        if f.segmentation_mask is None:
            return result
        person = np.squeeze(f.segmentation_mask) > 0.5
        if person.ndim != 2 or not person.any():
            return result
        mask_h, mask_w = person.shape

        # Height: silhouette crown-to-sole (includes hair and footwear), only when both ankles are genuinely
        # visible and the silhouette touches neither the top nor bottom edge (head and feet fully in frame)
        if self._ankles_in_frame(f):
            rows = np.flatnonzero(person.any(axis=1))
            top, bottom = int(rows[0]), int(rows[-1])
            if top > 0 and bottom < mask_h - 1:
                result["height"] = (bottom - top + 1) / mask_h * img_h * cm_per_px

        # Chest / waist: silhouette width at fixed fractions of the shoulder→hip span. Arms resting against the
        # torso merge into the silhouette; a torso is narrower than the shoulder-joint span, so a run much wider
        # than the shoulders means the arms are included and the value is dropped rather than shown wrong.
        sh_y = (l_sh[1] + r_sh[1]) / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        center_x = (l_sh[0] + r_sh[0] + l_hip[0] + r_hip[0]) / 4
        for key, frac in (("chest_width", CHEST_LINE_FRACTION), ("waist_width", WAIST_LINE_FRACTION)):
            run_px = self._silhouette_run_width(person, sh_y + frac * (hip_y - sh_y), center_x)
            if run_px is None:
                continue
            run_img_px = run_px / mask_w * img_w
            if run_img_px <= shoulder_px * ARM_MERGE_MAX_WIDTH_RATIO:
                result[key] = run_img_px * cm_per_px
        return result

    def _ankles_in_frame(self, f: FrameLandmarks) -> bool:
        """MediaPipe extrapolates off-screen ankles, so require real visibility and in-frame coordinates."""
        for name in ("left_ankle", "right_ankle"):
            pt = f.keypoints.get(name)
            if (
                not pt
                or f.visibility.get(name, 0.0) < ANKLE_VISIBILITY_MIN
                or not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0)
            ):
                return False
        return True

    def torso_fullness(self, f: FrameLandmarks) -> Optional[float]:
        """
        Mean midsection width with the arms excluded, over the shoulder-joint width, for one frame (scale-free,
        so a capture and a render can be compared directly). None without pose landmarks or a segmentation mask.
        """
        kp = f.keypoints
        l_sh, r_sh = kp.get("left_shoulder"), kp.get("right_shoulder")
        l_hip, r_hip = kp.get("left_hip"), kp.get("right_hip")
        if f.segmentation_mask is None or not (l_sh and r_sh and l_hip and r_hip):
            return None
        person = np.squeeze(f.segmentation_mask) > 0.5
        if person.ndim != 2 or not person.any():
            return None
        mask_h, mask_w = person.shape
        shoulder_px = math.hypot((r_sh[0] - l_sh[0]) * mask_w, (r_sh[1] - l_sh[1]) * mask_h)
        if shoulder_px < 1.0:
            return None

        sh_y = (l_sh[1] + r_sh[1]) / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        center_x = (l_sh[0] + r_sh[0] + l_hip[0] + r_hip[0]) / 4
        # Arm chains ordered image-left, image-right (MediaPipe's "left" is the person's left, i.e. image right)
        arms = sorted(
            ([l_sh, kp.get("left_elbow"), kp.get("left_wrist")], [r_sh, kp.get("right_elbow"), kp.get("right_wrist")]),
            key=lambda chain: chain[0][0],
        )
        widths = []
        for frac in MIDSECTION_LINE_FRACTIONS:
            y = sh_y + frac * (hip_y - sh_y)
            width = self._torso_run_width(person, y, center_x, [self._arm_center_x(a, y) for a in arms], shoulder_px)
            if width is not None:
                widths.append(width)
        if not widths:
            return None
        return float(np.mean(widths)) / shoulder_px

    @staticmethod
    def _arm_center_x(chain: List[Optional[Tuple[float, float, float]]], y_norm: float) -> Optional[float]:
        """x of the shoulder→elbow→wrist centre line at y_norm; below the last joint the forearm is taken to hang straight."""
        pts = [p for p in chain if p is not None]
        for a, b in zip(pts, pts[1:]):
            if a[1] <= y_norm <= b[1] and b[1] > a[1]:
                return a[0] + (y_norm - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
        if len(pts) >= 2 and y_norm > pts[-1][1]:
            return pts[-1][0]
        return None

    def _torso_run_width(
        self, person: np.ndarray, y_norm: float, x_norm: float, arm_xs: List[Optional[float]], shoulder_px: float
    ) -> Optional[float]:
        """
        Torso width (mask px) on one row. Arms hanging at the sides merge into the silhouette; where an arm's centre
        line lies inside the run, the arm's inner edge is its outer edge (the run end) mirrored about that centre line,
        and the torso is measured between the two inner edges. An arm outside the run is already separate.
        """
        run = self._silhouette_run(person, y_norm, x_norm)
        if run is None:
            return None
        scale = person.shape[1] - 1
        center = x_norm * scale
        edges = [float(run[0]), float(run[1])]
        for i, arm_x in enumerate(arm_xs):  # i=0 image-left arm / run start, i=1 image-right arm / run end
            if arm_x is None:
                continue
            arm_px, outer = arm_x * scale, edges[i]
            if not min(outer, center) < arm_px < max(outer, center):
                continue
            if abs(arm_px - outer) > ARM_HALF_WIDTH_MAX_RATIO * shoulder_px:
                return None
            edges[i] = 2 * arm_px - outer
        width = edges[1] - edges[0]
        return width if width >= TORSO_MIN_WIDTH_RATIO * shoulder_px else None

    @staticmethod
    def _silhouette_run(person: np.ndarray, y_norm: float, x_norm: float) -> Optional[Tuple[int, int]]:
        """(left, right) pixel columns of the contiguous silhouette run through (x_norm, y_norm); None if background."""
        h, w = person.shape
        y, x = int(round(y_norm * (h - 1))), int(round(x_norm * (w - 1)))
        if not (0 <= y < h and 0 <= x < w) or not person[y, x]:
            return None
        background = np.flatnonzero(~person[y])
        left_bg, right_bg = background[background < x], background[background > x]
        left = int(left_bg[-1]) + 1 if left_bg.size else 0
        right = int(right_bg[0]) - 1 if right_bg.size else w - 1
        return left, right

    @classmethod
    def _silhouette_run_width(cls, person: np.ndarray, y_norm: float, x_norm: float) -> Optional[int]:
        """Pixel length of the contiguous silhouette run through (x_norm, y_norm); None if that point is background."""
        run = cls._silhouette_run(person, y_norm, x_norm)
        return run[1] - run[0] + 1 if run else None

    def _classify_body_type(
        self,
        sh_to_hip: float,
        chest_prop: float,
        waist_prop: float,
        hip_prop: float
    ) -> str:
        if sh_to_hip > 1.20 and waist_prop < 0.88:
            return "Athletic / V-Shape"
        if chest_prop > 0.98 and waist_prop < 0.82 and 0.92 <= sh_to_hip <= 1.15:
            return "Hourglass / Curved"
        if sh_to_hip > 1.18:
            return "Inverted Triangle"
        if sh_to_hip < 0.95 or (hip_prop > 1.05 and sh_to_hip <= 1.0):
            return "Pear / Triangle"
        return "Rectangle / Balanced"


    def generate_face_and_hands_mask(
        self,
        image: Image.Image,
        feather_radius: int = 15,
        include_hands: bool = False,
        is_outerwear: bool = False,
        shirt_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates binary and feathered masks for protected regions.

        - Default (shirt_only=False): Masks face (and optionally hands) for identity protection.
        - shirt_only=True: Skips face entirely; only masks the inner shirt V-zone visible between
          blazer/suit lapels (chin to mid-torso, center strip). Use this for outerwear compositing
          so the user's original shirt color/pattern is restored without touching the face.
        - is_outerwear=True with shirt_only=False: Masks face AND inner shirt zone.

        Feathers boundary edges with a Gaussian gradient for seamless inpaint compositing.
        """
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        face_detected = False

        # 1. Face Detection via MediaPipe FaceLandmarker (478-point mesh)
        # Skipped in shirt_only mode — we only want the shirt zone mask, not the face.
        if not shirt_only and self._face_detector:
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
                with self.face_lock:
                    face_res = self._face_detector.detect(mp_img)
                if face_res and face_res.face_landmarks and len(face_res.face_landmarks) > 0:
                    fl = face_res.face_landmarks[0]
                    pts = np.array([[int(p.x * w), int(p.y * h)] for p in fl], dtype=np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.fillPoly(mask, [hull], 255)
                    face_detected = True
            except Exception as fe:
                print(f"[FACE MASK] FaceLandmarker warning: {fe}")

        # 2. Extract pose landmarks for hands and face fallback
        lm = self.extract_single_frame(image)
        if lm and lm.keypoints:
            kp = lm.keypoints
            # Fallback face detection if FaceLandmarker missed (skipped in shirt_only mode)
            if not shirt_only and not face_detected:
                nose = kp.get("nose")
                l_sh = kp.get("left_shoulder")
                r_sh = kp.get("right_shoulder")
                if nose and l_sh and r_sh:
                    nx, ny = int(nose[0] * w), int(nose[1] * h)
                    sh_w = int(abs(r_sh[0] - l_sh[0]) * w)
                    head_radius = max(25, int(sh_w * 0.38))
                    cv2.circle(mask, (nx, ny - int(head_radius * 0.2)), head_radius, 255, -1)
                    face_detected = True

            # Hands detection (optional; disabled by default to avoid spots from sleeve/pose shifts)
            if include_hands:
                l_sh = kp.get("left_shoulder")
                r_sh = kp.get("right_shoulder")
                l_hip = kp.get("left_hip")
                r_hip = kp.get("right_hip")

                torso_x_min = min(l_sh[0], r_sh[0]) - 0.03 if (l_sh and r_sh) else 0.25
                torso_x_max = max(l_sh[0], r_sh[0]) + 0.03 if (l_sh and r_sh) else 0.75
                torso_y_min = min(l_sh[1], r_sh[1]) - 0.02 if (l_sh and r_sh) else 0.25
                torso_y_max = max(l_hip[1], r_hip[1]) + 0.05 if (l_hip and r_hip) else 0.85

                for side in ["left", "right"]:
                    hand_pts = []
                    for pt_name in [f"{side}_wrist", f"{side}_pinky", f"{side}_index"]:
                        pt = kp.get(pt_name)
                        if pt:
                            # Reject keypoints falling inside the garment torso zone
                            if torso_x_min <= pt[0] <= torso_x_max and torso_y_min <= pt[1] <= torso_y_max:
                                continue
                            hand_pts.append([int(pt[0] * w), int(pt[1] * h)])

                    if hand_pts:
                        hand_r = max(10, int(w * 0.035))
                        for pt in hand_pts:
                            cv2.circle(mask, (pt[0], pt[1]), hand_r, 255, -1)
                        if len(hand_pts) >= 2:
                            hull_pts = cv2.convexHull(np.array(hand_pts, dtype=np.int32))
                            cv2.fillPoly(mask, [hull_pts], 255)

        # 3. Geometric fallback if neither face mesh nor pose detected (skipped in shirt_only mode)
        if not shirt_only and not face_detected:
            cv2.ellipse(mask, (w // 2, int(h * 0.12)), (int(w * 0.16), int(h * 0.12)), 0, 0, 360, 255, -1)

        binary_mask = (mask > 128).astype(np.uint8) * 255

        # Soft/feathered gradient: Gaussian blur at mask boundaries
        ksize = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
        feathered = cv2.GaussianBlur(binary_mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
        feathered_mask = np.clip(feathered, 0.0, 1.0)

        return binary_mask, feathered_mask


# Global singleton service
body_profile_service = BodyProfileService()
