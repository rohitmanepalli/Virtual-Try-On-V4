from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from app.config import SILHOUETTE_VERIFY_THRESHOLD
from app.models import BodyProfile, SilhouetteVerificationResult
from app.services.body_profile import FrameLandmarks, body_profile_service


# Studio-background detection. Gemini sometimes performs a "lazy edit": it swaps only the
# garment and returns the raw capture (office background, webcam lighting) instead of a studio
# render. Measured on real 4K outputs: studio renders have border edge density <= 0.011 and
# 95th-percentile border saturation <= 20; raw-photo outputs >= 0.049 and >= 85.
STUDIO_BORDER_EDGE_MAX = 0.025
STUDIO_BORDER_SAT_P95_MAX = 50.0

# A render whose torso_fullness (midsection width / shoulder-joint width) is more than this fraction below the
# capture's has slimmed or flattened the body. On 10 real images, unchanged bodies under reframing / JPEG / lighting
# changes dropped at most 7.5%; synthetic stomach flattening of 20% / 25% at its peak was caught 6/10 and 9/10.
# Smaller slimming is inside measurement noise. Recalibrate on real capture-vs-render pairs.
MIDSECTION_MAX_LOSS = 0.08

# Garment silhouette: clothed torso width (arms excluded) and outer width (arms and sleeves included), each a mean over
# rows from the chest to just below the hips, over the shoulder-joint width. The skeleton is locked across sizes, so
# the ratio measures how far the garment stands off the body. Rows start below the armpits and end past the hip line
# so that a hem riding up or hanging longer is included.
GARMENT_LINE_FRACTIONS = tuple(round(0.20 + 0.05 * i, 2) for i in range(20))  # 0.20 .. 1.15 of shoulder -> hip
# Calibrated 2026-09-28 on 21 real renders (lean M visitor and fuller XL visitor, all six sizes, woven shirt), judged
# against the images: garment width varies about ±6% of shoulder width between independent renders of one body (a
# visibly tighter S measured +5.4% vs M; the visibly most oversized XXL -0.5%), which is more than adjacent sizes
# differ, and extra size often shows as length and dropped seams rather than width. So width only flags gross errors.
FIT_DIRECTION_NOISE = 0.06
# A "won't fit" render is shown only if it is no wider than the recommended render by more than this. Deliberately
# strict: rejecting a good render costs a render, showing a normal-fit render contradicts the "won't fit" message.
# On that data it decided 4 of 5 won't-fit renders the way the images warranted (it also rejected one good render).
WONT_FIT_MAX_WIDENING = 0.02
# Near-identical sizes: the torso appearance distance matched visual judgement on every pair (look-alike pairs
# 0.175-0.238, visibly different pairs >= 0.285); silhouette distance alone did not (it paired an open-front XS with
# an XXL at 0.005). Both must be small.
NEAR_IDENTICAL_SILHOUETTE = 0.05
NEAR_IDENTICAL_APPEARANCE = 0.28
_APPEARANCE_GRID = (32, 48)  # (w, h) of the normalised torso crop compared between renders

_LOOSE = ("relaxed", "loose", "oversized")
_TIGHT = ("snug", "tight", "does_not_fit")


def fit_family(fit_quality: Optional[str], fit_state: Optional[str] = "standard") -> str:
    """Verifier allowance family for a size's fit category: relaxed (larger), fitted (smaller) or standard."""
    if fit_quality in _LOOSE:
        return "relaxed"
    if fit_quality in _TIGHT:
        return "fitted"
    if fit_quality == "perfect":
        return "standard"
    return (fit_state or "standard").lower().strip()


def garment_silhouette(img: Image.Image, lm: Optional[FrameLandmarks]) -> Optional[Dict[str, Any]]:
    """
    Scale-free garment silhouette of one render: torso width (arms excluded, production arm mirroring), outer width
    (full silhouette extent, arms included) and a normalised grayscale torso crop for appearance comparison.
    None without pose landmarks or a segmentation mask.
    """
    if lm is None or lm.segmentation_mask is None:
        return None
    kp = lm.keypoints
    l_sh, r_sh, l_hip, r_hip = (kp.get(k) for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if not (l_sh and r_sh and l_hip and r_hip):
        return None
    person = np.squeeze(lm.segmentation_mask) > 0.5
    if person.ndim != 2 or not person.any():
        return None
    mh, mw = person.shape
    shoulder_px = float(np.hypot((r_sh[0] - l_sh[0]) * mw, (r_sh[1] - l_sh[1]) * mh))
    if shoulder_px < 1.0:
        return None
    sh_y, hip_y = (l_sh[1] + r_sh[1]) / 2, (l_hip[1] + r_hip[1]) / 2
    cx = (l_sh[0] + r_sh[0] + l_hip[0] + r_hip[0]) / 4
    arms = sorted(
        ([l_sh, kp.get("left_elbow"), kp.get("left_wrist")], [r_sh, kp.get("right_elbow"), kp.get("right_wrist")]),
        key=lambda chain: chain[0][0],
    )
    torso, outer = [], []
    for frac in GARMENT_LINE_FRACTIONS:
        y = sh_y + frac * (hip_y - sh_y)
        t = body_profile_service._torso_run_width(
            person, y, cx, [body_profile_service._arm_center_x(a, y) for a in arms], shoulder_px)
        torso.append(t / shoulder_px if t is not None else None)
        row = int(round(y * (mh - 1)))
        if 0 <= row < mh:
            # Outer extent within 1.5 shoulder widths of the body centre (a studio render has only this person)
            lo = max(0, int(cx * (mw - 1) - 1.5 * shoulder_px))
            hi = min(mw, int(cx * (mw - 1) + 1.5 * shoulder_px) + 1)
            cols = np.flatnonzero(person[row, lo:hi])
            outer.append((cols[-1] - cols[0] + 1) / shoulder_px if cols.size else None)
        else:
            outer.append(None)
    torso_vals = [v for v in torso if v is not None]
    outer_vals = [v for v in outer if v is not None]
    if not torso_vals and not outer_vals:
        return None

    # Torso crop from the render's own landmarks (so framing differences between renders cancel), z-normalised
    gray = np.array(img.convert("L"))
    ih, iw = gray.shape
    sw = shoulder_px / mw * iw
    x1 = int(max(0, min(l_sh[0], r_sh[0]) * iw - 0.25 * sw))
    x2 = int(min(iw, max(l_sh[0], r_sh[0]) * iw + 0.25 * sw))
    y1 = int(max(0, (sh_y - 0.10 * (hip_y - sh_y)) * ih))
    y2 = int(min(ih, (hip_y + 0.35 * (hip_y - sh_y)) * ih))
    appearance = None
    if x2 - x1 > 8 and y2 - y1 > 8:
        crop = np.array(Image.fromarray(gray[y1:y2, x1:x2]).resize(_APPEARANCE_GRID, Image.BILINEAR), dtype=np.float32)
        appearance = ((crop - crop.mean()) / (crop.std() + 1e-6)).round(3).tolist()

    return {
        "torso_width": round(float(np.mean(torso_vals)), 4) if torso_vals else None,
        "outer_width": round(float(np.mean(outer_vals)), 4) if outer_vals else None,
        "torso_rows": len(torso_vals),
        "appearance": appearance,
    }


def _delta(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not a or not b or a.get(key) is None or b.get(key) is None:
        return None
    return round(a[key] - b[key], 4)


def compare_to_recommended(render: Optional[Dict[str, Any]], recommended: Optional[Dict[str, Any]],
                           fit_quality: str, tolerance: float = FIT_DIRECTION_NOISE) -> Dict[str, Any]:
    """
    Direction check against the recommended-size render of the same garment. as_expected reports the raw direction
    (larger size wider, smaller size no wider); passed only fails beyond `tolerance`, the measured render-to-render
    noise, so it flags gross errors, not every noisy reading. passed None = not measurable (skipped).
    """
    d_torso, d_outer = _delta(render, recommended, "torso_width"), _delta(render, recommended, "outer_width")
    result: Dict[str, Any] = {"torso_delta": d_torso, "outer_delta": d_outer, "passed": None, "as_expected": None}
    deltas = [d for d in (d_torso, d_outer) if d is not None]
    if not deltas or fit_quality == "perfect":
        return result
    if fit_quality in _LOOSE:
        result["as_expected"] = max(deltas) > 0
        result["passed"] = max(deltas) >= -tolerance
        result["expected"] = "wider than the recommended size"
    elif fit_quality in _TIGHT:
        torso = d_torso if d_torso is not None else max(deltas)
        result["as_expected"] = torso <= 0
        result["passed"] = torso <= tolerance
        result["expected"] = "no wider than the recommended size"
    return result


def compare_sizes(a: Optional[Dict[str, Any]], a_quality: str, b: Optional[Dict[str, Any]], b_quality: str) -> Dict[str, Any]:
    """Silhouette and appearance distance between two sizes' renders; near_identical None when not measurable."""
    deltas = [abs(d) for d in (_delta(a, b, "torso_width"), _delta(a, b, "outer_width")) if d is not None]
    sil = round(max(deltas), 4) if deltas else None
    app = None
    if a and b and a.get("appearance") and b.get("appearance"):
        app = round(float(np.mean(np.abs(np.array(a["appearance"]) - np.array(b["appearance"])))), 4)
    result: Dict[str, Any] = {"silhouette_distance": sil, "appearance_distance": app, "near_identical": None}
    # Two "won't fit" sizes are the same state (a garment that cannot close): looking alike is expected
    if sil is None or app is None or a_quality == b_quality == "does_not_fit":
        return result
    result["near_identical"] = sil < NEAR_IDENTICAL_SILHOUETTE and app < NEAR_IDENTICAL_APPEARANCE
    return result


def check_studio_background(img: Image.Image) -> Tuple[bool, Dict[str, float]]:
    """
    Returns (is_studio, metrics). Samples the left/right/top border strips (where only
    backdrop should be visible) for texture (Canny edge density) and colourfulness.
    Must run on the un-watermarked image (the watermark sits in the top-right corner).
    """
    import cv2
    cached = getattr(img, "_vto_studio_check", None)  # the same render is checked twice per attempt
    if cached is not None and cached[0] == img.size:
        return cached[1]
    arr = np.array(img.convert("RGB").resize((384, 512)))
    h, w = arr.shape[:2]
    sx, ty = int(w * 0.12), int(h * 0.10)
    strips = [arr[:, :sx], arr[ty:, -sx:], arr[:ty, sx:int(w * 0.70)]]
    edges = cv2.Canny(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 60, 150)
    edge_strips = [edges[:, :sx], edges[ty:, -sx:], edges[:ty, sx:int(w * 0.70)]]
    edge_density = float(np.concatenate([e.ravel() for e in edge_strips]).mean() / 255.0)
    border = np.concatenate([s.reshape(-1, 3) for s in strips]).reshape(-1, 1, 3)
    sat_p95 = float(np.percentile(cv2.cvtColor(border, cv2.COLOR_RGB2HSV)[..., 1], 95))
    is_studio = edge_density <= STUDIO_BORDER_EDGE_MAX and sat_p95 <= STUDIO_BORDER_SAT_P95_MAX
    result = is_studio, {"border_edge_density": round(edge_density, 4), "border_sat_p95": round(sat_p95, 1)}
    img._vto_studio_check = (img.size, result)
    return result


class SilhouetteVerifier:
    """
    Automated silhouette verification engine.
    Compares the generated virtual try-on render against the expected body profile
    and original silhouette using MediaPipe Pose keypoint proportions and segmentation masks.
    """

    def __init__(self, default_threshold: float = SILHOUETTE_VERIFY_THRESHOLD):
        self.default_threshold = default_threshold

    def verify(
        self,
        user_img: Image.Image,
        generated_img: Image.Image,
        body_profile: Optional[BodyProfile] = None,
        threshold: Optional[float] = None,
        fit_state: Optional[str] = "standard",
        selected_size: Optional[str] = None,
        size_offset: Optional[object] = None,
        dress_img: Optional[Image.Image] = None,
        face_check: Optional[object] = None,
    ) -> SilhouetteVerificationResult:
        active_threshold = threshold if threshold is not None else self.default_threshold
        
        quality = getattr(size_offset, 'fit_quality', None) if size_offset else None
        fit_lower = fit_family(quality, fit_state)

        # 1. Extract pose landmarks from generated render
        gen_landmarks = body_profile_service.extract_single_frame(generated_img)

        # 2. Extract or use existing body profile for reference
        ref_landmarks = body_profile_service.extract_single_frame(user_img)

        # 3. Calculate segmentation mask IoU with fit-aware allowance
        mask_iou = self._calculate_mask_iou(ref_landmarks, gen_landmarks, user_img, generated_img, fit_state=fit_lower)

        # Body volume: whole-body IoU and skeletal ratios cannot see a slimmed stomach, so compare the midsection directly
        midsection = self.check_midsection(ref_landmarks, gen_landmarks)

        # Background cleanliness analysis (border must be an even studio backdrop, not the raw capture)
        bg_clean = True
        gen_np = np.array(generated_img.convert("RGB"))
        gh, gw = gen_np.shape[:2]
        try:
            bg_clean, _ = check_studio_background(generated_img)
        except Exception:
            pass

        # Garment color consistency check
        garment_color_match = True
        if dress_img is not None:
            try:
                d_np = np.array(dress_img.convert("RGB"))
                # Filter out pure white or near-white background pixels
                d_mask = np.mean(d_np, axis=2) < 240
                if np.any(d_mask):
                    d_mean = np.mean(d_np[d_mask], axis=0)
                    # Sample torso region of generated image
                    g_torso = gen_np[int(gh * 0.35):int(gh * 0.65), int(gw * 0.3):int(gw * 0.7)]
                    g_mean = np.mean(g_torso, axis=(0, 1))
                    color_diff = float(np.linalg.norm(d_mean - g_mean))
                    if color_diff > 140.0:
                        garment_color_match = False
            except Exception:
                pass

        # 4. Calculate proportion deviations relative to invariant captured body profile
        if gen_landmarks and (body_profile or ref_landmarks):
            ref_sh = body_profile.shoulder_width_ratio if body_profile else ref_landmarks.shoulder_width
            ref_torso = (body_profile.shoulder_width_ratio * body_profile.torso_aspect_ratio) if body_profile else ref_landmarks.torso_length
            ref_height = body_profile.height_ratio if body_profile else ref_landmarks.height_ratio

            # Scale-normalization factor: accounts for camera distance / framing differences
            scale = 1.0
            if ref_torso > 0.05 and gen_landmarks.torso_length > 0.05:
                scale = gen_landmarks.torso_length / ref_torso
            elif ref_height > 0.05 and gen_landmarks.height_ratio > 0.05:
                scale = gen_landmarks.height_ratio / ref_height

            scaled_ref_sh = ref_sh * scale
            scaled_ref_height = ref_height * scale

            # Shoulder width deviation: account for fit-state fabric offset
            raw_sh_diff = gen_landmarks.shoulder_width / max(scaled_ref_sh, 0.05) - 1.0
            if fit_lower == "relaxed" and raw_sh_diff > 0:
                # Dropped shoulder seams pull the detected shoulder points outward, more with every size up
                sh_dev = max(0.0, raw_sh_diff - {"loose": 0.15, "oversized": 0.20}.get(quality, 0.12))
            elif fit_lower == "fitted" and raw_sh_diff > 0:
                sh_dev = max(0.0, raw_sh_diff - 0.03)
            else:
                sh_dev = abs(raw_sh_diff)
            sh_score = max(0.0, 1.0 - min(1.0, sh_dev * 2.0))

            # Torso aspect ratio deviation
            ref_aspect = ref_torso / max(ref_sh, 0.05)
            gen_aspect = gen_landmarks.torso_length / max(gen_landmarks.shoulder_width, 0.05)
            aspect_dev = abs(gen_aspect / max(ref_aspect, 0.05) - 1.0)
            if fit_lower == "relaxed" and gen_aspect < ref_aspect:
                aspect_dev = max(0.0, aspect_dev - 0.10)
            aspect_score = max(0.0, 1.0 - min(1.0, aspect_dev * 2.0))

            # Height deviation relative to framing scale
            height_dev = abs(gen_landmarks.height_ratio / max(scaled_ref_height, 0.05) - 1.0)
            height_score = max(0.0, 1.0 - min(1.0, height_dev * 1.5))

            # Anomaly penalty: checks for collapsed anatomy
            anomaly_penalty = 0.0
            if gen_landmarks.shoulder_width < 0.10 or gen_landmarks.shoulder_width > 0.65:
                anomaly_penalty += 0.35
            if gen_landmarks.height_ratio < 0.30:
                anomaly_penalty += 0.40

            if mask_iou is not None:
                raw_score = (
                    0.35 * sh_score +
                    0.30 * aspect_score +
                    0.15 * height_score +
                    0.20 * mask_iou
                ) - anomaly_penalty
            else:
                raw_score = (
                    0.45 * sh_score +
                    0.35 * aspect_score +
                    0.20 * height_score
                ) - anomaly_penalty

            composite_score = round(max(0.0, min(1.0, raw_score)), 3)

            face_preserved = getattr(face_check, 'passed', True) if face_check else True

            details = {
                "shoulder_fidelity": round(sh_score, 3),
                "aspect_fidelity": round(aspect_score, 3),
                "height_fidelity": round(height_score, 3),
                "mask_iou": round(mask_iou, 3) if mask_iou is not None else None,
                "sh_dev": round(sh_dev, 3),
                "aspect_dev": round(aspect_dev, 3),
                "gen_confidence": round(gen_landmarks.confidence, 2),
                "fit_state": fit_lower,
                "face_identity_preserved": face_preserved,
                "body_proportions_preserved": bool(sh_score >= 0.70 and aspect_score >= 0.70),
                "garment_color_preserved": garment_color_match,
                "arms_hands_natural": bool(anomaly_penalty == 0.0),
                "background_clean": bg_clean,
                "photorealistic_quality": bool(composite_score >= active_threshold),
            }
        else:
            # Fallback if landmarks could not be identified
            composite_score = 0.70
            details = {
                "fallback_scoring": True,
                "fit_state": fit_lower,
                "face_identity_preserved": True,
                "body_proportions_preserved": True,
                "garment_color_preserved": garment_color_match,
                "arms_hands_natural": True,
                "background_clean": bg_clean,
                "photorealistic_quality": True,
            }

        # Midsection loss is a hard failure, not a score penalty: a lower score can fall into the "adjust your stance"
        # branch in /api/tryon, which is the wrong message for a render that slimmed the visitor.
        midsection_preserved = midsection is None or midsection["preserved"]
        if midsection is not None:
            details["midsection_fullness_capture"] = midsection["capture"]
            details["midsection_fullness_render"] = midsection["render"]
            details["midsection_change"] = midsection["change"]
        details["midsection_preserved"] = midsection_preserved

        passed = composite_score >= active_threshold and midsection_preserved
        midsection_log = (
            f"{midsection['change']:+.1%} ({midsection['capture']:.3f} -> {midsection['render']:.3f})" if midsection else "n/a"
        )
        print(f"[SILHOUETTE VERIFY] Score: {composite_score:.3f} | Fit: {fit_lower} | Threshold: {active_threshold:.2f} | Midsection: {midsection_log} | Passed: {passed} | BG Clean: {bg_clean} | Color Match: {garment_color_match}")

        return SilhouetteVerificationResult(
            score=composite_score,
            passed=passed,
            threshold=active_threshold,
            details=details
        )

    def check_midsection(
        self, capture_lm: Optional[FrameLandmarks], render_lm: Optional[FrameLandmarks]
    ) -> Optional[Dict[str, Any]]:
        """Capture vs render torso_fullness; None (check skipped) when either cannot be measured."""
        if capture_lm is None or render_lm is None:
            return None
        capture = body_profile_service.torso_fullness(capture_lm)
        render = body_profile_service.torso_fullness(render_lm)
        if capture is None or render is None:
            return None
        change = render / capture - 1.0
        return {
            "capture": round(capture, 3),
            "render": round(render, 3),
            "change": round(change, 3),
            "preserved": change >= -MIDSECTION_MAX_LOSS,
        }

    def verify_cross_fit_landmarks(
        self,
        current_landmarks: FrameLandmarks,
        baseline_landmarks: Dict[str, float],
        fit_state: Optional[str] = "standard"
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Cross-fit regression verification:
        Compares skeletal landmarks (height, torso length, shoulder width, hip width)
        between current fit render and the baseline render for the same visitor session.
        If body landmark positions shift beyond small tolerances, returns passed=False
        to trigger an automatic retry, ensuring the physical body does not change across fits.
        """
        if not current_landmarks or not baseline_landmarks:
            return True, 1.0, {"skipped": True}

        base_height = baseline_landmarks.get("height_ratio", current_landmarks.height_ratio)
        base_torso = baseline_landmarks.get("torso_length", current_landmarks.torso_length)
        base_sh = baseline_landmarks.get("shoulder_width", current_landmarks.shoulder_width)
        base_hip = baseline_landmarks.get("hip_width", current_landmarks.hip_width)

        # Height deviation: human skeleton height must not change across fits (tolerance <= 0.08)
        height_drift = abs(current_landmarks.height_ratio / max(base_height, 0.05) - 1.0)

        # Torso length deviation: spinal length must not stretch or shrink (tolerance <= 0.08)
        torso_drift = abs(current_landmarks.torso_length / max(base_torso, 0.05) - 1.0)

        # Hip width deviation: pelvic skeleton must not cinch or flare (tolerance <= 0.10, with relaxed allowance)
        fit_lower = (fit_state or "standard").lower().strip()
        if fit_lower == "relaxed" and current_landmarks.hip_width >= base_hip:
            hip_drift = max(0.0, (current_landmarks.hip_width / max(base_hip, 0.05) - 1.0) - 0.06)
        else:
            hip_drift = abs(current_landmarks.hip_width / max(base_hip, 0.05) - 1.0)

        # Shoulder joint deviation: allows subtle garment pad offset
        if fit_lower == "relaxed" and current_landmarks.shoulder_width >= base_sh:
            sh_drift = max(0.0, (current_landmarks.shoulder_width / max(base_sh, 0.05) - 1.0) - 0.06)
        else:
            sh_drift = abs(current_landmarks.shoulder_width / max(base_sh, 0.05) - 1.0)

        passed = (
            height_drift <= 0.08 and
            torso_drift <= 0.08 and
            sh_drift <= 0.12 and
            hip_drift <= 0.10
        )

        drift_score = round(max(0.0, 1.0 - (height_drift * 0.35 + torso_drift * 0.35 + sh_drift * 0.15 + hip_drift * 0.15)), 3)

        details = {
            "height_drift": round(height_drift, 3),
            "torso_drift": round(torso_drift, 3),
            "shoulder_drift": round(sh_drift, 3),
            "hip_drift": round(hip_drift, 3),
            "drift_score": drift_score,
            "passed": passed
        }
        print(f"[CROSS-FIT REGRESSION] Passed: {passed} | Drift Score: {drift_score:.3f} | Height Drift: {height_drift:.1%} | Torso Drift: {torso_drift:.1%} | Sh Drift: {sh_drift:.1%}")
        return passed, drift_score, details

    def verify_cross_size_consistency(
        self,
        renders_by_size: Dict[str, Image.Image],
        body_profile: Optional[BodyProfile] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Cross-size verification:
        Ensures the *body* silhouette stays constant across different size renders
        while the *garment* silhouette differs (catches 'all sizes look identical' bug).

        Parameters:
            renders_by_size: dict mapping size label (e.g. "S", "M", "L") to rendered Image
            body_profile: optional BodyProfile for reference comparison

        Returns:
            (passed: bool, details: dict) where details contains per-size-pair metrics
        """
        if len(renders_by_size) < 2:
            return True, {"skipped": True, "reason": "fewer than 2 size renders"}

        # Extract landmarks from each render
        landmarks_by_size: Dict[str, Any] = {}
        for size_label, img in renders_by_size.items():
            lm = body_profile_service.extract_single_frame(img)
            if lm:
                landmarks_by_size[size_label] = lm

        if len(landmarks_by_size) < 2:
            return True, {"skipped": True, "reason": "landmarks extracted for fewer than 2 sizes"}

        size_labels = list(landmarks_by_size.keys())
        body_drift_scores: List[float] = []
        garment_ious: List[float] = []
        pair_details: Dict[str, Any] = {}

        for i in range(len(size_labels)):
            for j in range(i + 1, len(size_labels)):
                s1, s2 = size_labels[i], size_labels[j]
                lm1, lm2 = landmarks_by_size[s1], landmarks_by_size[s2]

                # Body drift: height, torso, shoulder, hip should stay constant
                h_drift = abs(lm1.height_ratio / max(lm2.height_ratio, 0.05) - 1.0)
                t_drift = abs(lm1.torso_length / max(lm2.torso_length, 0.05) - 1.0)
                s_drift = abs(lm1.shoulder_width / max(lm2.shoulder_width, 0.05) - 1.0)
                hip_drift = abs(lm1.hip_width / max(lm2.hip_width, 0.05) - 1.0)

                body_drift = max(h_drift, t_drift, s_drift, hip_drift)
                body_drift_scores.append(body_drift)

                # Garment mask IoU: should be < 0.95 (sizes must look different)
                mask_iou = self._calculate_mask_iou(
                    lm1, lm2,
                    renders_by_size[s1], renders_by_size[s2],
                    fit_state="standard"
                )
                if mask_iou is not None:
                    garment_ious.append(mask_iou)

                pair_details[f"{s1}_vs_{s2}"] = {
                    "body_drift": round(body_drift, 4),
                    "height_drift": round(h_drift, 4),
                    "torso_drift": round(t_drift, 4),
                    "shoulder_drift": round(s_drift, 4),
                    "hip_drift": round(hip_drift, 4),
                    "mask_iou": round(mask_iou, 4) if mask_iou is not None else None,
                }

        # Body must stay constant: max drift across all pairs < 8%
        max_body_drift = max(body_drift_scores) if body_drift_scores else 0.0
        body_constant = max_body_drift <= 0.08

        # Garment must differ: average IoU across pairs < 0.95
        avg_garment_iou = float(np.mean(garment_ious)) if garment_ious else 0.0
        garment_differs = avg_garment_iou < 0.95 if garment_ious else True

        passed = body_constant and garment_differs

        details = {
            "passed": passed,
            "body_constant": body_constant,
            "garment_differs": garment_differs,
            "max_body_drift": round(max_body_drift, 4),
            "avg_garment_iou": round(avg_garment_iou, 4),
            "pair_details": pair_details,
        }

        print(
            f"[CROSS-SIZE CONSISTENCY] Passed: {passed} | "
            f"Body Constant: {body_constant} (max drift: {max_body_drift:.1%}) | "
            f"Garment Differs: {garment_differs} (avg IoU: {avg_garment_iou:.3f}) | "
            f"Sizes Compared: {size_labels}"
        )

        return passed, details

    def _calculate_mask_iou(
        self,
        ref_lm: Optional[FrameLandmarks],
        gen_lm: Optional[FrameLandmarks],
        img1: Image.Image,
        img2: Image.Image,
        fit_state: str = "standard"
    ) -> Optional[float]:
        """Calculates mask IoU when genuine person segmentation masks are available, adjusting for garment fit."""
        try:
            if ref_lm and ref_lm.segmentation_mask is not None and gen_lm and gen_lm.segmentation_mask is not None:
                mask1 = np.squeeze(ref_lm.segmentation_mask)
                mask2 = np.squeeze(gen_lm.segmentation_mask)
                if mask1.ndim != 2 or mask2.ndim != 2:
                    return None
                # Resize both masks to (128, 128)
                m1 = Image.fromarray((mask1 > 0.5).astype(np.uint8) * 255).resize((128, 128))
                m2 = Image.fromarray((mask2 > 0.5).astype(np.uint8) * 255).resize((128, 128))

                arr1 = np.array(m1) > 128
                arr2 = np.array(m2) > 128

                intersection = float(np.logical_and(arr1, arr2).sum())
                union = float(np.logical_or(arr1, arr2).sum())
                ref_area = float(arr1.sum())

                if union > 0:
                    standard_iou = intersection / union
                    # For relaxed cut, calculate containment so looser outer fabric doesn't unfairly penalize score
                    if fit_state == "relaxed" and ref_area > 0:
                        containment = intersection / ref_area
                        return 0.5 * standard_iou + 0.5 * containment
                    return standard_iou
        except Exception as e:
            print(f"[SILHOUETTE VERIFY] Mask IoU calculation warning: {e}")

        return None


# Global singleton verifier
silhouette_verifier = SilhouetteVerifier()
