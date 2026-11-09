"""
Virtual compact powder: local, deterministic complexion finish on the visitor's own photo (no image model).

Where: facial skin only. The face-mesh oval (pulled in from the hairline and temples, but carried across the jawline
into the neck and faded out there, so no edge marks where the powder stops), minus eyes, eyebrows and lips, and minus
everything that does not look like this visitor's own skin. The skin reference is measured on both cheeks and the
forehead; beard, moustache, stubble, sideburns, hair, nostrils and deep shadow are much darker or a different colour
than it, so they get no powder (no dusty residue on facial hair).

What, in CIELAB, split into scales so pores and fine texture are never smoothed away:
- evens out blotchiness: the skin's low-frequency colour is pulled toward a broader average;
- controls shine: low-frequency lightness above the skin reference (T-zone glare, oily sheen) is pressed down;
- tints: a shaded powder nudges the skin toward its colour; translucent powder adds no colour.

Finishes: matte presses shine down hardest; natural velvet evens and softly mattifies; luminous keeps a soft glow,
lifting the face's own high points (forehead, cheekbones, bridge of the nose) instead of flattening them.

The powdered skin covers opacity(intensity) of the bare skin (beauty_tuning, capped at POWDER_OPACITY_MAX), mixed in
linear light. All look tuning values live in beauty_tuning.
"""
from typing import Optional

import cv2
import numpy as np

from app.models import CompactPowderFinish
from app.services.beauty_tuning import (EVEN_SIGMA_FRAC, FEATURE_MARGIN_FRAC, GLOW_SPAN, JAW_EXTEND_FRAC,
                                        JAW_FEATHER_FRAC, OVAL_FEATHER_FRAC, OVAL_INSET_FRAC, POWDER_FINISHES,
                                        POWDER_OPACITY_MAX, POWDER_OPACITY_MIN, TEXTURE_SIGMA_FRAC, TINT_CHROMA,
                                        TINT_LIGHTNESS, TINT_LIGHTNESS_MAX, opacity)
from app.services.face_preservation import BROWS, EYES
from app.services.lipstick_service import FACE_OVAL, LIPS_OUTER, MESH_POINTS, composite, shade_lab, smooth_contour

LEFT_EYE_ZONE = EYES[:16] + BROWS[:10]
RIGHT_EYE_ZONE = EYES[16:] + BROWS[10:]
SKIN_SAMPLES = (50, 280, 151)  # left cheek, right cheek, forehead
FACE_WIDTH_POINTS = (234, 454)
CHIN = 152

SAMPLE_RADIUS_FRAC = 0.04
# Skin likeness: pixels darker than the reference by more than DARK_START L* fade out of the mask by DARK_START+DARK_SPAN
DARK_START, DARK_SPAN = 10.0, 10.0
CHROMA_START, CHROMA_SPAN = 9.0, 9.0  # same for a*/b* distance (hair, lips, glasses frames)


def _fill(shape, poly: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    return mask


def _hull(shape, pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pts).astype(np.int32)), 255)
    return mask


def _masked_blur(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(img * weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return num / np.maximum(den, 1e-4)


def face_width(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts[FACE_WIDTH_POINTS[0]] - pts[FACE_WIDTH_POINTS[1]]))


def build_skin_mask(lab: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Float32 0-1: facial skin that receives powder (no eyes, brows, lips, facial hair or hair)."""
    shape = lab.shape[:2]
    fw = face_width(pts)
    oval = _fill(shape, smooth_contour(pts[FACE_OVAL]))
    # Signed distance to the face outline: + inside, - outside
    d = cv2.distanceTransform(oval, cv2.DIST_L2, 5) - cv2.distanceTransform(255 - oval, cv2.DIST_L2, 5)
    # Where the powder ends, by height: above the cheekbones it stops inside the outline (off the hairline); from there
    # down to the chin that edge moves out past the jawline and softens, so the powder fades into the neck instead
    # of ending in a line along the jaw.
    cheek_y = float(pts[list(FACE_WIDTH_POINTS), 1].mean())
    w = np.clip((np.arange(shape[0], dtype=np.float32)[:, None] - cheek_y) / max(float(pts[CHIN, 1]) - cheek_y, 1.0),
                0.0, 1.0)
    edge = fw * (OVAL_INSET_FRAC * (1.0 - w) - JAW_EXTEND_FRAC * w)
    feather = np.maximum(fw * (OVAL_FEATHER_FRAC * (1.0 - w) + JAW_FEATHER_FRAC * w), 1.0)
    region = np.clip((d - edge) / feather, 0.0, 1.0)

    features = _hull(shape, pts[LEFT_EYE_ZONE]) | _hull(shape, pts[RIGHT_EYE_ZONE]) | _fill(shape, smooth_contour(pts[LIPS_OUTER]))
    d_feat = cv2.distanceTransform(255 - features, cv2.DIST_L2, 5)
    margin = fw * FEATURE_MARGIN_FRAC
    region *= np.clip((d_feat - margin) / max(margin, 1.0), 0.0, 1.0)

    # Skin reference from cheek and forehead patches
    r = max(2, int(fw * SAMPLE_RADIUS_FRAC))
    samples = []
    for idx in SKIN_SAMPLES:
        x, y = np.round(pts[idx]).astype(int)
        patch = lab[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3)
        if patch.size:
            samples.append(patch)
    if not samples:
        return np.zeros(shape, dtype=np.float32)
    ref = np.median(np.concatenate(samples), axis=0)

    def likeness_to(ref_L, ref_a, ref_b, dark_start):
        darker = np.maximum(ref_L - lab[..., 0], 0.0)
        chroma = np.hypot(lab[..., 1] - ref_a, lab[..., 2] - ref_b)
        return (np.clip(1.0 - (darker - dark_start) / DARK_SPAN, 0.0, 1.0)
                * np.clip(1.0 - (chroma - CHROMA_START) / CHROMA_SPAN, 0.0, 1.0))

    # Pass 1, face-wide reference with a loose darkness limit: side lighting makes one half of the face darker, so
    # lightness is only trusted locally. Pass 2 compares each pixel to the skin around it: facial hair is darker than
    # its neighbouring skin, a shaded cheek is not.
    rough = likeness_to(ref[0], ref[1], ref[2], DARK_START + 20.0) * (region > 0)
    sigma = max(3.0, fw * 0.10)
    local_ref = [_masked_blur(lab[..., c], rough, sigma) for c in range(3)]
    likeness = likeness_to(local_ref[0], local_ref[1], local_ref[2], DARK_START)
    # Close the gaps between beard hairs too, so no powder lands in stubble
    likeness = cv2.erode(likeness, np.ones((3, 3), np.uint8))
    likeness = cv2.GaussianBlur(likeness, (0, 0), sigmaX=max(0.8, fw * 0.004))
    return (region * likeness).astype(np.float32)


def apply_powder(rgb: np.ndarray, pts: np.ndarray, hex_color: Optional[str] = None, intensity: float = 0.6,
                 finish: CompactPowderFinish = CompactPowderFinish.NATURAL_VELVET) -> np.ndarray:
    """Applies compact powder (hex_color None = translucent) to facial skin. Returns a new RGB uint8 image.

    `intensity` is the position on the powder curve (0-1, see beauty_tuning.POWDER_COVERAGE_T); 0 = untouched.
    """
    prof = POWDER_FINISHES[finish]
    if len(pts) < MESH_POINTS:
        raise ValueError("Powder placement needs a full face mesh")
    out = np.array(rgb[..., :3], dtype=np.uint8, copy=True)
    layer_opacity = opacity(intensity, POWDER_OPACITY_MIN, POWDER_OPACITY_MAX)
    if layer_opacity <= 0.0:
        return out
    h, w = out.shape[:2]
    oval = pts[FACE_OVAL]
    pad = 4 + face_width(pts) * (JAW_EXTEND_FRAC + JAW_FEATHER_FRAC)  # room for the fade past the jawline
    x0, y0 = np.floor(np.maximum(oval.min(axis=0) - pad, 0)).astype(int)
    x1, y1 = np.ceil(np.minimum(oval.max(axis=0) + pad, [w, h])).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out

    local = (pts - np.array([x0, y0], dtype=np.float32)).astype(np.float32)
    roi = out[y0:y1, x0:x1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    mask = build_skin_mask(lab, local)
    hard = (mask > 0.5).astype(np.float32)
    if hard.sum() < 50:
        return out

    fw = face_width(local)
    low = np.dstack([_masked_blur(lab[..., c], hard, max(1.0, fw * TEXTURE_SIGMA_FRAC)) for c in range(3)])
    texture = lab - low
    broad = np.dstack([_masked_blur(lab[..., c], hard, max(2.0, fw * EVEN_SIGMA_FRAC)) for c in range(3)])
    new_low = low + (broad - low) * prof.even_strength

    ref_L = float(np.median(new_low[..., 0][hard > 0]))
    above = new_low[..., 0] - ref_L
    excess = np.maximum(above - prof.shine_margin, 0.0)
    new_low[..., 0] -= excess * prof.shine_strength
    if prof.glow > 0:
        # Broad and soft: measured on the broad average, so it follows the face's planes, never single pores
        lift = np.clip((broad[..., 0] - ref_L) / GLOW_SPAN, 0.0, 1.0)
        new_low[..., 0] += prof.glow * (0.35 + 0.65 * lift)

    if hex_color:
        # One shift for the whole face, measured from its median skin: pulling each pixel toward the shade would
        # flatten the face's own shading and undertone variation (cheek warmth, contours) into a painted, mask-like look
        t = shade_lab(hex_color)
        ref = np.median(new_low[hard > 0], axis=0)
        new_low[..., 0] += float(np.clip((t[0] - ref[0]) * TINT_LIGHTNESS, -TINT_LIGHTNESS_MAX, TINT_LIGHTNESS_MAX))
        new_low[..., 1] += float((t[1] - ref[1]) * TINT_CHROMA)
        new_low[..., 2] += float((t[2] - ref[2]) * TINT_CHROMA)

    powdered = new_low + texture * prof.texture_keep
    powdered[..., 0] = np.clip(powdered[..., 0], 0.0, 100.0)
    colored = np.clip(cv2.cvtColor(powdered.astype(np.float32), cv2.COLOR_LAB2RGB), 0.0, 1.0)
    # The only place opacity is applied: skin mask x the coverage's point on the curve, once, in linear light
    out[y0:y1, x0:x1] = np.round(composite(roi, colored, mask * layer_opacity) * 255.0).astype(np.uint8)
    return out
