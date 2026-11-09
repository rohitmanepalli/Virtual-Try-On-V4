"""
Face preservation: the visitor's real captured face is the only face in the final image.

The image model regenerates the whole frame (including the face) no matter what the prompt
says, so identity can only be guaranteed at pixel level. `restore_original_face()` is the ONE
face-compositing path in the application:

1. MediaPipe face-mesh landmarks are detected in the capture and in the render.
2. A similarity transform (uniform scale + rotation + translation, never a stretch) is estimated
   from rigid landmarks (eyes, nose, forehead, temples) with RANSAC and then validated: inlier
   ratio, scale vs. face-size ratio, rotation, residual error, and face-shape agreement.
3. The capture is warped ONCE with Lanczos onto the render's own canvas (no intermediate resizes).
4. Inside the real face outline every pixel comes from the capture, after ONE global colour adjustment
   (lightness, contrast and a capped white-balance shift, CIELAB) measured on the skin of both faces, so a
   dim, green-cast webcam face matches the studio-lit render. The only mixing is a thin band on the outer
   perimeter of the face (skin meeting skin), plus a low-frequency lighting match that fades to zero well
   before any facial feature. There is no translucent full-face blend, no rectangle, no fixed ellipse,
   and no "paste at the frame position" fallback. If the pasted face's colour still differs visibly from
   the render's face, the paste is dropped and the clean render is served instead.
5. Before compositing, the render's own eyes, brows, nose, mouth and glasses zone must lie fully
   inside the opaque interior. If they would not (the model drew a differently placed/shaped
   face), nothing is composited and the render is reported as NOT restored so the caller can
   regenerate instead of serving two faces.

`verify_face_preserved()` independently re-detects the face in the finished output and
compares it with the capture, so a broken or skipped composite is caught and logged.
"""
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

# Both must hold for a face to count as preserved. A regenerated AI face of the same
# person scores SSIM ~0.35-0.75; a composited real face ~0.95+ (measured on real 4K renders).
FACE_PIXEL_SIM_THRESHOLD = 0.94
FACE_SSIM_THRESHOLD = 0.90

# MediaPipe face-mesh (478 points) index groups
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
# Rigid skull anchors: eye corners, nose bridge -> tip, forehead midline, temples (no mouth/jaw: expression moves them)
RIGID_ANCHORS = [33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 1, 10, 151, 9, 234, 454]
BROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MESH_POINTS = 400  # at least this many points means a full MediaPipe face mesh

# Alignment validation (fractions are of the face width in the render)
ALIGN_RANSAC_THRESH_FRAC = 0.03
ALIGN_MIN_INLIER_RATIO = 0.60
ALIGN_MAX_ROTATION_DEG = 20.0
ALIGN_SCALE_TOLERANCE = 0.20  # transform scale vs. face-width ratio
ALIGN_MAX_ANCHOR_RMS_FRAC = 0.04
SHAPE_MAX_RMS_FRAC = 0.07  # face outline agreement after alignment

# Compositing geometry (fractions of the face width in the render)
FEATHER_FRAC = 0.03  # perimeter transition band, inside the real face outline
TONE_RING_FRAC = 0.10  # perimeter ring used to measure the lighting difference
TONE_SIGMA_FRAC = 0.06  # low-pass scale of the lighting correction
TONE_DEPTH_FRAC = 0.20  # lighting correction fades to zero this far inside the outline
FEATURE_MARGIN_FRAC = 0.02  # margin around eyes/brows/nose/lips that must be fully opaque
GLASSES_MARGIN_FRAC = 0.07  # extra margin around eyes+brows covering spectacle frames
MAX_FEATURE_LEAK = 0.005  # max share of the render's feature zone allowed outside the opaque interior
# Max skin/beard mismatch at the seam after lighting match. Calibrated on 72 capture->render pairs of real
# 3584x4800 renders: same-face renders <= 0.070; renders where the model drew a different face 0.075-0.098.
SEAM_MAX_RESIDUAL = 0.075
# Max enlargement of the capture face onto the render. Beyond this the capture has too little facial detail:
# the transplanted face is a soft/blocky patch inside the sharp render (the visible "face mask" regression,
# e.g. a ~25px webcam face stretched 30x onto a 726px render face). Measured on real renders: 2.6x clean,
# 3.9x acceptable, 6.2x visibly soft patch. Above the limit the render is served uncomposited (its own
# single, clean face) instead of a low-detail real face.
MAX_FACE_UPSCALE = 4.0
# Colour match of the real face to the render (CIELAB, measured on cheek/forehead skin of both faces, features masked).
# Kiosk webcam faces are far darker and greener than studio-lit renders (297 captures: median L 124 / a* 136 vs ~180 /
# ~145 in 8-bit LAB); pasted unchanged they showed as a dark olive face on a bright body.
TONE_MAX_CHROMA_SHIFT = 15.0  # max a*/b* (white balance) correction (kiosk captures need ~8-11); a bigger cast is not trusted
TONE_CONTRAST_RANGE = (0.75, 1.33)  # allowed lightness contrast scaling
TONE_MIN_SKIN_PX = 200  # fewer measurable skin pixels than this: no colour match
# After compositing, the real face's skin must be within this CIELAB distance (dE*ab) of the render's own face skin;
# otherwise the paste is dropped and the clean render is served (an off-colour face is worse than the model's face).
MAX_FACE_TONE_DELTA_E = 5.0  # matched faces measure 0.5-2; a visible yellow-green residue measured 4.7


@dataclass
class FaceCheck:
    pixel_similarity: float
    ssim: float
    passed: bool
    reason: str = ""
    restored: bool = False  # True only when the real face was actually composited onto the render
    details: Dict[str, Any] = field(default_factory=dict)
    # True when compositing was deliberately skipped because the capture face cannot be pasted cleanly: too small
    # to enlarge (> MAX_FACE_UPSCALE), or its colour cannot be matched to the render (> MAX_FACE_TONE_DELTA_E).
    # The render is safe to serve as-is (one clean face, no patch). False + restored=False means "do not serve".
    low_detail_skip: bool = False

    @property
    def score(self) -> float:
        """Conservative single score reported to clients: the weaker of the two metrics."""
        return round(min(self.pixel_similarity, self.ssim), 4)


@dataclass
class Alignment:
    M: Optional[np.ndarray]
    ok: bool
    reason: str = ""
    scale: float = 0.0
    rotation_deg: float = 0.0
    inlier_ratio: float = 0.0
    anchor_rms_frac: float = 0.0
    shape_rms_frac: float = 0.0


def _to_rgb_np(img) -> np.ndarray:
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    return np.asarray(img)[..., :3]


# MediaPipe resizes the face crop to 192px internally, so detecting on a <=1600px copy of a 4K frame costs
# no landmark precision but avoids handing it ~17 megapixels. Returned points are in full-resolution pixels.
DETECT_MAX_SIDE = 1600


def detect_face_points(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Returns the 478 MediaPipe face-mesh points (pixel coords) of the most central face, or None.

    The face detector sees the whole frame at a small input size, so in a full-length render (a long outfit framed
    head to ankle) a clearly visible face can be too small for it. A miss is retried on the top half of the frame,
    where a standing person's head is; the crop starts at row 0, so its points are already in frame coordinates.
    """
    from app.services.body_profile import body_profile_service
    detector = getattr(body_profile_service, "_face_detector", None)
    if detector is None:
        return None
    pts = _detect_mesh(detector, rgb)
    if pts is None:
        pts = _detect_mesh(detector, rgb[: rgb.shape[0] // 2])
    return pts


def _detect_mesh(detector: Any, rgb: np.ndarray) -> Optional[np.ndarray]:
    import mediapipe as mp
    import traceback
    h, w = rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / float(max(h, w)))
    small = rgb if scale >= 1.0 else cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    from app.services.body_profile import body_profile_service
    try:
        with body_profile_service.face_lock:
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
    except Exception as err:
        print(f"[FACE PRESERVE] Face detection error: {err}")
        traceback.print_exc()
        return None
    if not res or not res.face_landmarks:
        return None
    fl = min(res.face_landmarks, key=lambda f: abs(np.mean([p.x for p in f]) - 0.5))
    return np.array([[p.x * w, p.y * h] for p in fl], dtype=np.float32)


def _dilate_disc(mask: np.ndarray, radius: float) -> np.ndarray:
    """Binary dilation by a disc of `radius` px (distance transform: exact and fast for large radii)."""
    if radius <= 0:
        return mask
    dist_out = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(dist_out <= radius, 255, 0).astype(np.uint8)


def _is_mesh(pts: np.ndarray) -> bool:
    return len(pts) >= MESH_POINTS


def _outline(pts: np.ndarray) -> np.ndarray:
    return pts[FACE_OVAL] if _is_mesh(pts) else pts


def _anchors(pts: np.ndarray) -> np.ndarray:
    return pts[RIGID_ANCHORS] if _is_mesh(pts) else pts


def _face_width(pts: np.ndarray) -> float:
    if _is_mesh(pts):
        return float(np.linalg.norm(pts[234] - pts[454]))
    return float(np.ptp(pts[:, 0]))


def _feature_groups(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(all inner features, eyes+brows for the glasses zone). Generic point sets use a shrunken copy."""
    if _is_mesh(pts):
        return pts[BROWS + EYES + NOSE + LIPS], pts[BROWS + EYES]
    c = pts.mean(0)
    inner = c + (pts - c) * 0.55
    return inner, inner


def _transform(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def _frame_mapping(src_shape, dst_shape) -> np.ndarray:
    """Plain scale mapping from the capture frame to the output frame. Used by verification only."""
    sy = dst_shape[0] / float(src_shape[0])
    sx = dst_shape[1] / float(src_shape[1])
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float32)


def estimate_alignment(pts_src: Optional[np.ndarray], pts_dst: Optional[np.ndarray]) -> Alignment:
    """
    Robust similarity transform capture-face -> render-face, validated before use.
    Rejects (ok=False) rather than returning a guess: an unvalidated transform is what places a
    second face beside the model's face.
    """
    if pts_src is None or pts_dst is None:
        return Alignment(None, False, "face landmarks missing")
    if len(pts_src) != len(pts_dst) or len(pts_src) < 10:
        return Alignment(None, False, "landmark sets incompatible")

    src, dst = _anchors(pts_src), _anchors(pts_dst)
    face_w_dst = _face_width(pts_dst)
    face_w_src = _face_width(pts_src)
    if face_w_dst < 8 or face_w_src < 4:
        return Alignment(None, False, "face too small")

    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=max(2.0, ALIGN_RANSAC_THRESH_FRAC * face_w_dst),
        maxIters=2000, confidence=0.995,
    )
    if M is None or inliers is None:
        return Alignment(None, False, "no transform found")

    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    inlier_mask = inliers.ravel().astype(bool)
    inlier_ratio = float(inlier_mask.mean())
    expected_scale = face_w_dst / face_w_src
    anchor_err = np.linalg.norm(_transform(src, M) - dst, axis=1)
    anchor_rms = float(np.sqrt(np.mean(anchor_err[inlier_mask] ** 2))) / face_w_dst if inlier_mask.any() else 1.0
    shape_err = np.linalg.norm(_transform(_outline(pts_src), M) - _outline(pts_dst), axis=1)
    shape_rms = float(np.sqrt(np.mean(shape_err ** 2))) / face_w_dst

    result = Alignment(M.astype(np.float32), True, "", scale, rotation, inlier_ratio, anchor_rms, shape_rms)
    if inlier_ratio < ALIGN_MIN_INLIER_RATIO:
        result.ok, result.reason = False, f"too few consistent landmarks ({inlier_ratio:.0%})"
    elif abs(scale / expected_scale - 1.0) > ALIGN_SCALE_TOLERANCE:
        result.ok, result.reason = False, f"scale {scale:.2f} inconsistent with face size ratio {expected_scale:.2f}"
    elif abs(rotation) > ALIGN_MAX_ROTATION_DEG:
        result.ok, result.reason = False, f"rotation {rotation:.1f} deg too large"
    elif anchor_rms > ALIGN_MAX_ANCHOR_RMS_FRAC:
        result.ok, result.reason = False, f"landmark residual {anchor_rms:.3f} too large"
    elif shape_rms > SHAPE_MAX_RMS_FRAC:
        result.ok, result.reason = False, f"render face shape differs from capture ({shape_rms:.3f})"
    return result


def _align(pts_src: Optional[np.ndarray], pts_dst: Optional[np.ndarray], src_shape, dst_shape) -> Tuple[np.ndarray, str]:
    """Verification helper: validated landmark transform, else the plain frame mapping (never used to composite)."""
    alignment = estimate_alignment(pts_src, pts_dst)
    if alignment.ok:
        return alignment.M, "landmarks"
    return _frame_mapping(src_shape, dst_shape), "frame"


@dataclass
class RestorePlan:
    """Where and how the real face is transplanted, in render-canvas coordinates (internal math only)."""

    roi: Tuple[int, int, int, int]  # x1, y1, x2, y2 on the render
    outline: np.ndarray  # warped real face outline, render coords
    real_glasses_zone: np.ndarray  # warped real eyes+brows (spectacles reach the outline at the temples)
    face_w: float  # face width on the render, px
    gen_features: np.ndarray  # render's own inner features, render coords
    gen_glasses_zone: np.ndarray  # render's own eyes+brows, render coords
    layer: Optional["FaceLayer"] = None  # filled once by restore_original_face


def _plan(pts_o: np.ndarray, pts_g: np.ndarray, M: np.ndarray, gen_shape) -> RestorePlan:
    H, W = gen_shape[:2]
    outline = _transform(_outline(pts_o), M)
    real_glasses = _transform(_feature_groups(pts_o)[1], M)
    face_w = _face_width(pts_g)
    features, glasses = _feature_groups(pts_g)
    pad = face_w * 0.25
    x1, y1 = np.floor(np.minimum(outline.min(0), features.min(0)) - pad).astype(int)
    x2, y2 = np.ceil(np.maximum(outline.max(0), features.max(0)) + pad).astype(int)
    roi = (max(0, x1), max(0, y1), min(W, x2), min(H, y2))
    return RestorePlan(roi, outline, real_glasses, face_w, features, glasses)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _region_alpha(plan: RestorePlan, shape) -> Tuple[np.ndarray, np.ndarray]:
    """
    (alpha, distance-inside) for the ROI: alpha is exactly 1 inside the real face beyond the thin perimeter band.
    The real face region is the face outline plus the real glasses zone widened by the band, so frames that
    reach the temples are wholly real and the band never crosses a spectacle frame.
    """
    x1, y1 = plan.roi[:2]
    feather = max(2.0, FEATHER_FRAC * plan.face_w)
    region = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(region, [cv2.convexHull((plan.outline - [x1, y1]).astype(np.int32))], 255)
    glasses = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(glasses, [cv2.convexHull((plan.real_glasses_zone - [x1, y1]).astype(np.int32))], 255)
    region |= _dilate_disc(glasses, GLASSES_MARGIN_FRAC * plan.face_w + feather)
    dist_in = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    alpha = _smoothstep(dist_in / feather)
    alpha[dist_in >= feather] = 1.0
    alpha[region == 0] = 0.0
    return alpha.astype(np.float32), dist_in


def _feature_zone(plan: RestorePlan, shape, with_real_glasses: bool = False) -> np.ndarray:
    """The render's own eyes/brows/nose/mouth and glasses zone (optionally also the real one), widened, in the ROI."""
    x1, y1 = plan.roi[:2]
    zone = np.zeros(shape, dtype=np.uint8)
    groups = [(plan.gen_features, FEATURE_MARGIN_FRAC), (plan.gen_glasses_zone, GLASSES_MARGIN_FRAC)]
    if with_real_glasses:
        groups.append((plan.real_glasses_zone, GLASSES_MARGIN_FRAC))
    for pts, margin in groups:
        part = np.zeros_like(zone)
        cv2.fillPoly(part, [cv2.convexHull((pts - [x1, y1]).astype(np.int32))], 255)
        zone |= _dilate_disc(part, margin * plan.face_w)
    return zone > 0


def _feature_leak(plan: RestorePlan, alpha: np.ndarray) -> float:
    """Share of the render's own feature zone (incl. glasses) that would NOT be fully replaced."""
    inside = _feature_zone(plan, alpha.shape)
    if not inside.any():
        return 1.0
    return float(np.mean(alpha[inside] < 0.999))


@dataclass
class Tone:
    """
    One global colour adjustment of the real face, in CIELAB (L 0-100): L' = (L - l_src) * l_scale + l_dst,
    a' = a + a_shift, b' = b + b_shift. The default is the identity.
    """

    l_src: float = 0.0
    l_dst: float = 0.0
    l_scale: float = 1.0
    a_shift: float = 0.0
    b_shift: float = 0.0

    def as_details(self) -> Dict[str, float]:
        return {k: round(float(v), 3) for k, v in vars(self).items()}


def _lab(rgb: np.ndarray) -> np.ndarray:
    """float CIELAB (L 0-100, a/b about -127..127) from RGB 0-255."""
    return cv2.cvtColor((np.asarray(rgb, np.float32) / 255.0).astype(np.float32), cv2.COLOR_RGB2LAB)


def _apply_tone(rgb: np.ndarray, tone: Tone) -> np.ndarray:
    """The tone adjustment applied to every pixel; float32 RGB 0-255 out."""
    lab = _lab(rgb)
    lab[..., 0] = np.clip((lab[..., 0] - tone.l_src) * tone.l_scale + tone.l_dst, 0.0, 100.0)
    lab[..., 1] += tone.a_shift
    lab[..., 2] += tone.b_shift
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255.0, 0.0, 255.0)


def _skin_lab(rgb: np.ndarray, skin: np.ndarray) -> Optional[np.ndarray]:
    """CIELAB of the skin pixels, without the darkest/brightest 5% (hair strands, shadows, highlights)."""
    if int(skin.sum()) < TONE_MIN_SKIN_PX:
        return None
    lab = _lab(rgb)[skin]
    lo, hi = np.percentile(lab[:, 0], [5, 95])
    return lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]


def _estimate_tone(real: np.ndarray, render: np.ndarray, skin: np.ndarray) -> Tone:
    """Colour adjustment that gives the real face's skin the render's lighting (identity when skin is not measurable)."""
    src, dst = _skin_lab(real, skin), _skin_lab(render, skin)
    if src is None or dst is None or len(src) < TONE_MIN_SKIN_PX // 2 or len(dst) < TONE_MIN_SKIN_PX // 2:
        return Tone()
    scale = float(np.clip(dst[:, 0].std() / max(float(src[:, 0].std()), 1e-3), *TONE_CONTRAST_RANGE))
    shift = np.clip(dst[:, 1:].mean(0) - src[:, 1:].mean(0), -TONE_MAX_CHROMA_SHIFT, TONE_MAX_CHROMA_SHIFT)
    return Tone(float(src[:, 0].mean()), float(dst[:, 0].mean()), scale, float(shift[0]), float(shift[1]))


def _skin_delta_e(a_rgb: np.ndarray, b_rgb: np.ndarray, skin: np.ndarray) -> float:
    """CIELAB distance (dE*ab) between the mean skin colours of two aligned images (0 when skin is not measurable)."""
    a, b = _skin_lab(a_rgb, skin), _skin_lab(b_rgb, skin)
    if a is None or b is None or not len(a) or not len(b):
        return 0.0
    return float(np.linalg.norm(a.mean(0) - b.mean(0)))


@dataclass
class FaceLayer:
    """The real-face layer for the ROI, computed once per restore (internal math only)."""

    real_face: np.ndarray  # float32 RGB: warped capture, colour-matched, + perimeter lighting match
    gen_roi: np.ndarray  # float32 RGB: the render inside the ROI
    alpha: np.ndarray  # float32: 1 inside the real face, thin perimeter ramp, 0 outside
    seam_residual: float  # structure mismatch between real and rendered skin at the seam (0-1)
    tone: Tone = field(default_factory=Tone)  # the global colour adjustment applied to the real face
    skin: Optional[np.ndarray] = None  # bool, ROI: face skin (interior, features and glasses excluded)


def _low_pass(img: np.ndarray, weights: np.ndarray, sigma: float, face_w: float) -> np.ndarray:
    """
    Normalized Gaussian average of `img` over `weights`, computed on a ~96px-per-face downsample
    (the result is deliberately low-frequency, so full-resolution blurring would only cost time).
    """
    h, w = weights.shape
    k = max(1.0, face_w / 96.0)
    sw, sh = max(8, int(round(w / k))), max(8, int(round(h / k)))
    small_w = cv2.resize(weights, (sw, sh), interpolation=cv2.INTER_AREA)
    small_i = cv2.resize(img * weights[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    s = max(0.5, sigma * sw / float(w))
    den = cv2.GaussianBlur(small_w, (0, 0), s)
    num = cv2.GaussianBlur(small_i, (0, 0), s)
    field = num / np.maximum(den, 1e-3)[..., None]
    field[den < 1e-3] = 0.0
    return cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)


def _face_layer(
    orig_rgb: np.ndarray, gen_rgb: np.ndarray, plan: RestorePlan, M: np.ndarray,
    alpha_dist: Optional[Tuple[np.ndarray, np.ndarray]] = None, keep_capture_tone: bool = False,
) -> FaceLayer:
    x1, y1, x2, y2 = plan.roi
    rw, rh = x2 - x1, y2 - y1
    M_roi = M.astype(np.float64).copy()
    M_roi[:, 2] -= [x1, y1]
    # The capture's pixels, resampled ONCE (Lanczos) straight onto the render canvas
    warped = cv2.warpAffine(orig_rgb, M_roi, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    warped = warped.astype(np.float32)
    gen_roi = gen_rgb[y1:y2, x1:x2].astype(np.float32)

    alpha, dist_in = alpha_dist if alpha_dist is not None else _region_alpha(plan, (rh, rw))

    # One global colour adjustment (CIELAB lightness, contrast and capped white balance) measured on the skin of
    # both faces: a dim, green-cast webcam face gets the studio-lit render's tone. Every pixel gets the same
    # mapping, so the features stay the capture's own (relit) pixels, with no render detail mixed in.
    # keep_capture_tone (beauty): the skin tone is what the visitor is judging makeup against, so the face keeps its
    # own colour; if the render's skin is too far from it, the tone gate in restore_original_face refuses the paste.
    skin = (alpha >= 0.999) & ~_feature_zone(plan, alpha.shape, with_real_glasses=True)
    tone = Tone() if keep_capture_tone else _estimate_tone(warped, gen_roi, skin)
    warped = _apply_tone(warped, tone)

    # Low-frequency lighting match measured on the perimeter ring only, so the real face meets the
    # studio-lit render without a visible edge. It carries no facial detail (heavy low-pass) and fades
    # to exactly zero before the features, which therefore stay the capture's own pixels.
    ring = ((dist_in > 0) & (dist_in <= TONE_RING_FRAC * plan.face_w)).astype(np.float32)
    lighting = _low_pass(gen_roi - warped, ring, TONE_SIGMA_FRAC * plan.face_w, plan.face_w)
    weight = 1.0 - _smoothstep(dist_in / max(1.0, TONE_DEPTH_FRAC * plan.face_w))
    real_face = warped + lighting * weight[..., None]

    # What remains different at the seam after lighting is matched: beard vs. bare skin, different jaw, etc.
    ring_b = ring.astype(bool)
    luma = np.float32([0.299, 0.587, 0.114])
    if ring_b.any():
        seam = float(np.mean(np.abs((gen_roi - (warped + lighting)) @ luma)[ring_b])) / 255.0
    else:
        seam = 1.0
    return FaceLayer(real_face, gen_roi, alpha, seam, tone, skin)


def _composite(orig_rgb: np.ndarray, gen_rgb: np.ndarray, plan: RestorePlan, M: np.ndarray) -> np.ndarray:
    """
    Transplants the real face onto the render (returns a new array; inputs untouched).

    Inside the face outline beyond the thin perimeter band, pixels are the capture's own pixels
    (single Lanczos warp, one global colour adjustment; the lighting match is zero there). The perimeter band cross-fades real
    skin into rendered skin; no facial feature is ever inside it (checked by _feature_leak) and the
    skin on both sides must agree (checked by the seam residual).
    """
    layer = getattr(plan, "layer", None) or _face_layer(orig_rgb, gen_rgb, plan, M)
    x1, y1, x2, y2 = plan.roi
    a = layer.alpha[..., None]
    out = gen_rgb.copy()
    out[y1:y2, x1:x2] = np.clip(layer.real_face * a + layer.gen_roi * (1.0 - a), 0, 255).astype(np.uint8)
    return out


def check_facial_anomaly(img_rgb: np.ndarray, pts: Optional[np.ndarray]) -> bool:
    """
    Detects if the composited face region exhibits unnatural artifacts,
    such as flat/uniform color blocks or zero variance.
    Returns True if an anomaly is detected (face is broken).
    """
    if pts is None or len(pts) < 10:
        return False
    xs, ys = pts[:, 0].astype(int), pts[:, 1].astype(int)
    h, w = img_rgb.shape[:2]
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    rw, rh = int(np.ptp(xs) * 0.25), int(np.ptp(ys) * 0.25)
    y1, y2 = max(0, cy - rh), min(h, cy + rh)
    x1, x2 = max(0, cx - rw), min(w, cx + rw)
    patch = img_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    std_val = float(np.mean(np.std(patch.astype(float), axis=(0, 1))))
    return std_val < 5.0


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    s_a = blur(a * a) - mu_a ** 2
    s_b = blur(b * b) - mu_b ** 2
    s_ab = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))


def verify_face_preserved(original_img, output_img) -> FaceCheck:
    """
    Independently re-detects the face in the finished output, maps it back onto the
    capture at the capture's native resolution, refines the registration to sub-pixel
    accuracy (ECC), and compares the inner face region (pixel similarity + SSIM).

    Comparing at native resolution matters: landmark jitter of a few pixels on a 4K
    output would otherwise depress SSIM for a genuinely preserved face.
    """
    orig = _to_rgb_np(original_img)
    out = _to_rgb_np(output_img)
    pts_o = detect_face_points(orig)
    if pts_o is None:
        return FaceCheck(0.0, 0.0, False, "no face detected in capture")
    pts_out = detect_face_points(out)
    if pts_out is None:
        return FaceCheck(0.0, 0.0, False, "no face detected in output")

    M, _ = _align(pts_o, pts_out, orig.shape, out.shape)
    h, w = orig.shape[:2]
    # Downsample the output to capture scale first (area filter), then undo the alignment.
    s = float(np.hypot(M[0, 0], M[1, 0]))
    if s > 1.0:
        out = cv2.resize(out, (max(1, round(out.shape[1] / s)), max(1, round(out.shape[0] / s))), interpolation=cv2.INTER_AREA)
        M = M / s
    back = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

    region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(region, [cv2.convexHull(pts_o.astype(np.int32))], 1)
    face_w = max(1.0, float(np.sqrt(region.sum())))
    k = 2 * max(2, int(face_w * 0.12)) + 1
    core = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if not np.any(core):
        return FaceCheck(0.0, 0.0, False, "face region too small to verify")

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    back_gray = cv2.cvtColor(back, cv2.COLOR_RGB2GRAY)
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        _, warp = cv2.findTransformECC(orig_gray, back_gray, warp, cv2.MOTION_EUCLIDEAN, criteria, region, 5)
        # Only accept small corrections; a large one means ECC locked onto something else.
        if np.abs(warp[:, 2]).max() < 0.05 * face_w:
            back = cv2.warpAffine(back, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
            back_gray = cv2.cvtColor(back, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        pass

    core = core.astype(bool)
    # Illumination calibration: prevents overall studio lighting/exposure differences
    # from falsely penalizing facial identity similarity
    lum_delta = float(orig[core].astype(np.float32).mean() - back[core].astype(np.float32).mean())
    norm_back = np.clip(back[core].astype(np.float32) + lum_delta, 0.0, 255.0)
    mae = float(np.mean(np.abs(orig[core].astype(np.float32) - norm_back)))
    pixel_sim = max(0.0, 1.0 - mae / 255.0)
    ssim = float(np.mean(_ssim_map(orig_gray, back_gray)[core]))

    passed = pixel_sim >= FACE_PIXEL_SIM_THRESHOLD and ssim >= FACE_SSIM_THRESHOLD
    return FaceCheck(round(pixel_sim, 4), round(ssim, 4), passed, "" if passed else "below threshold")


def log_face_check(check: FaceCheck, context: str) -> None:
    print(
        f"[FACE CHECK TEMPORARY LOG] {context}: "
        f"passed={check.passed}, "
        f"pixel={check.pixel_similarity:.3f} (threshold={FACE_PIXEL_SIM_THRESHOLD}), "
        f"ssim={check.ssim:.3f} (threshold={FACE_SSIM_THRESHOLD}), "
        f"reason='{check.reason}'"
    )
    if check.passed:
        print(f"[FACE CHECK] {context}: preserved (pixel={check.pixel_similarity:.3f}, ssim={check.ssim:.3f})")
    else:
        print(
            f"[FACE REGRESSION] {context}: output face does NOT match the captured face "
            f"(pixel={check.pixel_similarity:.3f} < {FACE_PIXEL_SIM_THRESHOLD} or ssim={check.ssim:.3f} < {FACE_SSIM_THRESHOLD}; {check.reason})"
        )


def _encode_lossless(rgb: np.ndarray) -> bytes:
    """
    Lossless intermediate: the single lossy encode happens once, when the final result is delivered.
    Uncompressed (BMP) because PNG-compressing a 3584x4800 frame takes seconds on kiosk CPUs while an
    uncompressed container is bit-exact and near-instant; it never leaves the server.
    """
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="BMP")
    return buf.getvalue()


def _not_restored(orig: np.ndarray, generated_bytes: bytes, gen: np.ndarray, reason: str, **details) -> Tuple[bytes, FaceCheck]:
    check = verify_face_preserved(orig, gen)
    check.restored = False
    check.passed = False
    check.reason = f"real face NOT restored: {reason}"
    check.details = details
    log_face_check(check, "restore rejected (render left uncomposited; caller must not serve it)")
    return generated_bytes, check


def restore_original_face(
    original_img: Image.Image,
    generated_bytes: bytes,
    fallback_mask: Optional[np.ndarray] = None,
    keep_capture_tone: bool = False,
    max_upscale: float = MAX_FACE_UPSCALE,
) -> Tuple[bytes, FaceCheck]:
    """
    Replaces the model's face with the visitor's real captured face (the single face-compositing path).

    Returns (image_bytes, FaceCheck). When check.restored is True the bytes are a lossless image at
    the render's full resolution containing exactly one face: the real one. When restoration is
    impossible or unsafe (no face in capture or render, invalid alignment, the model's facial
    features would stay visible) the ORIGINAL render bytes are returned untouched with
    restored=False, and must not be served as a finished try-on. The one exception is
    low_detail_skip=True: the capture face is too small to enlarge (> MAX_FACE_UPSCALE) without
    becoming a visible soft patch, so the untouched render (one clean face) is safe to serve.

    `fallback_mask` is accepted for API compatibility and deliberately unused: a mask not derived
    from both faces' landmarks is exactly what used to paste a second face in the wrong place.

    `keep_capture_tone` keeps the real face's own skin colour instead of matching it to the render's lighting (beauty:
    a render that lightened the skin is refused by the tone gate, never copied onto the visitor). `max_upscale` lowers
    the enlargement limit (beauty: an enlarged face has lost the fine detail makeup is judged on).
    """
    orig = _to_rgb_np(original_img)
    gen = _to_rgb_np(generated_bytes)
    base_log = {"input_size": f"{orig.shape[1]}x{orig.shape[0]}", "generated_size": f"{gen.shape[1]}x{gen.shape[0]}"}

    pts_o = detect_face_points(orig)
    if pts_o is None:
        _log_restore(base_log, transform_valid=False, face_restore_applied=False, reason="no face in capture")
        return _not_restored(orig, generated_bytes, gen, "no face detected in capture")
    pts_g = detect_face_points(gen)
    if pts_g is None:
        _log_restore(base_log, transform_valid=False, face_restore_applied=False, reason="no face in render")
        return _not_restored(orig, generated_bytes, gen, "no face detected in render")

    gx1, gy1 = pts_g.min(0)
    gx2, gy2 = pts_g.max(0)
    base_log["face_bbox"] = [int(gx1), int(gy1), int(gx2), int(gy2)]
    capture_face_px, render_face_px = _face_width(pts_o), _face_width(pts_g)
    upscale = render_face_px / max(capture_face_px, 1e-6)
    base_log.update(capture_face_px=round(capture_face_px), render_face_px=round(render_face_px), upscale=round(upscale, 2))

    # Decided before alignment: a tiny face often cannot even be aligned, and must never become a blurry patch
    if upscale > max_upscale:
        _log_restore(base_log, transform_valid=None, face_restore_applied=False,
                     reason=f"capture face too small ({capture_face_px:.0f}px, x{upscale:.1f} > x{max_upscale:.1f}); render served uncomposited")
        check = verify_face_preserved(orig, gen)
        check.restored = False
        check.passed = False
        check.low_detail_skip = True
        check.reason = "capture face too small to restore without a visible low-detail patch"
        check.details = {"capture_face_px": round(capture_face_px, 1), "render_face_px": round(render_face_px, 1),
                         "upscale": round(upscale, 2), "canvas": [int(gen.shape[1]), int(gen.shape[0])]}
        return generated_bytes, check

    alignment = estimate_alignment(pts_o, pts_g)
    details = {
        "scale": round(alignment.scale, 3),
        "rotation_deg": round(alignment.rotation_deg, 2),
        "inlier_ratio": round(alignment.inlier_ratio, 3),
        "anchor_rms_frac": round(alignment.anchor_rms_frac, 4),
        "shape_rms_frac": round(alignment.shape_rms_frac, 4),
        "capture_face_px": round(capture_face_px, 1),
        "render_face_px": round(render_face_px, 1),
        "canvas": [int(gen.shape[1]), int(gen.shape[0])],
    }
    if not alignment.ok:
        _log_restore(base_log, transform_valid=False, face_restore_applied=False, reason=alignment.reason)
        return _not_restored(orig, generated_bytes, gen, f"alignment rejected: {alignment.reason}", **details)

    plan = _plan(pts_o, pts_g, alignment.M, gen.shape)
    x1, y1, x2, y2 = plan.roi
    alpha_dist = _region_alpha(plan, (y2 - y1, x2 - x1))
    leak = _feature_leak(plan, alpha_dist[0])
    details["feature_leak"] = round(leak, 4)
    if leak > MAX_FEATURE_LEAK:
        _log_restore(base_log, transform_valid=True, face_restore_applied=False, reason=f"feature leak {leak:.4f}")
        return _not_restored(
            orig, generated_bytes, gen,
            f"render's own eyes/brows/nose/mouth/glasses would remain visible ({leak:.1%} outside the real face)", **details,
        )

    plan.layer = _face_layer(orig, gen, plan, alignment.M, alpha_dist, keep_capture_tone)
    details["seam_residual"] = round(plan.layer.seam_residual, 4)
    if plan.layer.seam_residual > SEAM_MAX_RESIDUAL:
        _log_restore(base_log, transform_valid=True, face_restore_applied=False, reason=f"seam {plan.layer.seam_residual:.4f}")
        return _not_restored(
            orig, generated_bytes, gen,
            f"render's face surround differs from the real face (seam {plan.layer.seam_residual:.3f}); the join would be visible",
            **details,
        )

    comp = _composite(orig, gen, plan, alignment.M)

    # Safety gate on the finished pixels: the real face's skin must now have the render's face tone. If it does not
    # (a colour cast too strong to correct), a dark or green-tinted face would sit on the studio-lit body, which is
    # worse than the model's own face: the paste is dropped and the clean render is served (as for a tiny face).
    tone_de = _skin_delta_e(comp[y1:y2, x1:x2], gen[y1:y2, x1:x2], plan.layer.skin)
    details["tone"] = plan.layer.tone.as_details()
    details["tone_delta_e"] = round(tone_de, 2)
    if tone_de > MAX_FACE_TONE_DELTA_E:
        _log_restore(base_log, transform_valid=True, face_restore_applied=False,
                     reason=f"face colour dE {tone_de:.1f} > {MAX_FACE_TONE_DELTA_E:.0f}; render served uncomposited")
        check = verify_face_preserved(orig, gen)
        check.restored = False
        check.passed = False
        check.low_detail_skip = True
        check.reason = (f"real face colour could not be matched to the render's lighting (dE {tone_de:.1f} > "
                        f"{MAX_FACE_TONE_DELTA_E:.0f})")
        check.details = details
        return generated_bytes, check

    # Verified against the capture with the same colour adjustment: the check is about WHOSE face it is (position,
    # features, no blend), not about keeping the webcam's lighting. (A very dark capture, L* < ~30, relit this far
    # can score SSIM ~0.87 for a correct face: its 8-bit noise is amplified. The score is reported, not a serve gate.)
    relit_orig = np.clip(_apply_tone(orig, plan.layer.tone) + 0.5, 0, 255).astype(np.uint8)
    check = verify_face_preserved(relit_orig, comp)
    check.restored = True
    check.details = details
    if check_facial_anomaly(comp, pts_g):
        print("[FACE ANOMALY] Flat/uniform block artifact detected in composite.")
        check.passed = False
        check.reason = "facial anomaly: uniform color block detected"

    detail_note = (
        f"capture face {details['capture_face_px']:.0f}px -> {details['render_face_px']:.0f}px on "
        f"{gen.shape[1]}x{gen.shape[0]} canvas (x{alignment.scale:.1f} Lanczos)"
    )
    _log_restore(base_log, transform_valid=True, face_restore_applied=True,
                 working_resolution=f"{gen.shape[1]}x{gen.shape[0]}", quality_check="pass" if check.passed else "below threshold")
    log_face_check(check, f"real face restored ({detail_note})")
    return _encode_lossless(comp), check


def _log_restore(base: Dict[str, Any], **fields) -> None:
    """One structured diagnostic line per restore attempt. Geometry and sizes only: never pixels or images."""
    items = {**base, **fields}
    print("FACE_RESTORE: " + " ".join(f"{k}={v}" for k, v in items.items()), flush=True)
