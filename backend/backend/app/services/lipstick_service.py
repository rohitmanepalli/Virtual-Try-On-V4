"""
Virtual lipstick: local, deterministic lip recolouring on the visitor's own photo (no image model, no API cost).

1. MediaPipe face-mesh (this module's own FaceLandmarker) gives the outer and inner lip contours. Each contour is
   resampled as a smooth closed spline, so the lip line is a clean curve (Cupid's bow included), not a polygon.
2. Lip mask from distance fields: the outer edge is inset slightly (the landmarks sit a hair outside the vermilion
   border) with a 1-3 px anti-aliased falloff that thins out further into the mouth corners; the mouth opening gets
   a very short falloff, so teeth and tongue are never tinted. Small mouths are drawn supersampled (smooth edges).
3. Colour in CIELAB, split into scales: the lip's own base colour is replaced by the shade, mixed back a little
   toward the lip's natural tone (more for a shade lighter than the lip), and the lip's broad 3D shading and fine
   texture (creases) are put back on top, so it reads as lipstick on these lips, not a sticker.
4. The finish decides how much of that comes back and what light does on it:
   - matte:  flat and velvety. Texture kept, shading compressed, every highlight removed, slightly deeper colour.
   - cream:  soft satin. More of the lip's shading, plus a broad, diffuse sheen on the lower lip.
   - gloss:  wet. Creases partly filled in, and sharp, white specular highlights: on the lower lip, on both peaks of
             the upper lip, and wherever the photo's own light already hits the lip.
5. That lipstick layer covers opacity(intensity) of the lip (beauty_tuning: a curve, not linear), mixed with the
   bare lip in linear light (composite).

Only a padded crop around the mouth is processed, so cost is independent of the photo's resolution.
All look tuning values live in beauty_tuning.
"""
import os
import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from app.models import LipstickFinish
from app.services.beauty_tuning import (CHROMA_DETAIL, CLOSED_GAP_FRAC, CORNER_FADE_DEPTH, CORNER_FADE_RADIUS_FRAC,
                                        EDGE_FEATHER_FRAC, EDGE_FEATHER_RANGE, EDGE_INSET_FRAC, FINISHES,
                                        INNER_FEATHER_FRAC, INNER_FEATHER_RANGE, LIP_OPACITY_MAX, LIP_OPACITY_MIN,
                                        MASK_SUPERSAMPLE_MOUTH_PX, NATURAL_TONE_BASE, NATURAL_TONE_LIGHT,
                                        NATURAL_TONE_LIGHT_SPAN, SCENE_LIGHT_GAINS, SHADING_SIGMA_FRAC, FinishProfile,
                                        opacity)

# MediaPipe face-mesh (478 points) lip contours, each ordered as a closed loop
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
MOUTH_CORNERS = (61, 291)
UPPER_OUTER_MID, UPPER_INNER_MID = 0, 13
LOWER_INNER_MID, LOWER_OUTER_MID = 14, 17
UPPER_PEAKS = ((37, 82), (267, 312))  # (outer, inner) point pairs under the two peaks of the Cupid's bow
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
MESH_POINTS = 468  # FaceLandmarker returns 478 (468 mesh + 10 iris); fewer means it is not a face mesh

# Below this mouth width (px) there is too little lip to recolour believably: a full-length capture from 2.5 m.
MIN_MOUTH_WIDTH_PX = 18.0
MIN_FACE_WIDTH_PX = 60.0  # same for a powder-only look (cheek to cheek)
ROI_PAD_FRAC = 0.30  # crop padding around the lips
SPLINE_SAMPLES = 12  # spline points per landmark segment


class LipsNotFound(Exception):
    """No usable lips in the photo. The message is safe to show to the visitor."""


@dataclass
class LipstickResult:
    image: np.ndarray  # RGB uint8, same size as the input
    face_box: Dict[str, float]  # normalized x, y, w, h
    mouth_width_px: float


def hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected a #RRGGBB colour, got {hex_color!r}")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def mouth_width(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]]))


def face_box(pts: np.ndarray, width: int, height: int) -> Dict[str, float]:
    oval = pts[FACE_OVAL]
    x0, y0 = np.clip(oval.min(axis=0), 0, [width, height])
    x1, y1 = np.clip(oval.max(axis=0), 0, [width, height])
    return {"x": float(x0 / width), "y": float(y0 / height), "w": float((x1 - x0) / width), "h": float((y1 - y0) / height)}


def smooth_contour(poly: np.ndarray, samples: int = SPLINE_SAMPLES) -> np.ndarray:
    """Closed centripetal-free (uniform) Catmull-Rom spline through the points: a smooth lip line."""
    p = np.asarray(poly, dtype=np.float32)
    p0, p1, p2, p3 = np.roll(p, 1, axis=0), p, np.roll(p, -1, axis=0), np.roll(p, -2, axis=0)
    t = np.linspace(0.0, 1.0, samples, endpoint=False, dtype=np.float32)[:, None, None]
    t2, t3 = t * t, t * t * t
    curve = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    return curve.transpose(1, 0, 2).reshape(-1, 2)  # segment-major order


def _fill(shape, poly: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    return mask


def build_lip_mask(shape, pts: np.ndarray, _px: float = 1.0) -> np.ndarray:
    """Float32 0-1 lip mask: clean anti-aliased outer edge inset slightly and thinning into the mouth corners, mouth
    opening (teeth/tongue) at exactly 0. `_px` is internal: mask pixels per photo pixel when drawn supersampled."""
    mw = mouth_width(pts)
    s = int(np.clip(np.ceil(MASK_SUPERSAMPLE_MOUTH_PX / max(mw, 1.0)), 1, 4)) if _px == 1.0 else 1
    if s > 1:
        # Small mouth: a pixel-grid polygon would give a stair-stepped lip line, so draw the mask s times larger
        # (pixel centres map as (p + 0.5) * s - 0.5) and area-average it back down: sub-pixel smooth edges.
        big = build_lip_mask((shape[0] * s, shape[1] * s), (pts + 0.5) * s - 0.5, _px=float(s))
        return cv2.resize(big, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    outer = _fill(shape, smooth_contour(pts[LIPS_OUTER]))
    inner = _fill(shape, smooth_contour(pts[LIPS_INNER]))
    inset = mw * EDGE_INSET_FRAC
    feather = float(np.clip(mw * EDGE_FEATHER_FRAC, EDGE_FEATHER_RANGE[0] * _px, EDGE_FEATHER_RANGE[1] * _px))
    inner_feather = float(np.clip(mw * INNER_FEATHER_FRAC, INNER_FEATHER_RANGE[0] * _px, INNER_FEATHER_RANGE[1] * _px))
    # Distance (px) inside the outer contour, and outside the mouth opening; +0.5 centres the pixel on the edge
    d_outer = cv2.distanceTransform(outer, cv2.DIST_L2, 5) - 0.5
    d_inner = cv2.distanceTransform(255 - inner, cv2.DIST_L2, 5) - 0.5
    a_outer = np.clip((d_outer - inset) / feather, 0.0, 1.0)
    a_inner = np.clip(d_inner / inner_feather, 0.0, 1.0)
    a_outer = a_outer * a_outer * (3.0 - 2.0 * a_outer)  # smoothstep: soft, but no visible halo
    # Real lipstick thins out into the mouth corners: fade to (1 - CORNER_FADE_DEPTH) at each corner point
    r = mw * CORNER_FADE_RADIUS_FRAC
    for corner in pts[list(MOUTH_CORNERS)]:
        a_outer = a_outer * (1.0 - CORNER_FADE_DEPTH * _gaussian_spot(shape, corner, r, r))
    if float(np.linalg.norm(pts[UPPER_INNER_MID] - pts[LOWER_INNER_MID])) < mw * CLOSED_GAP_FRAC:
        # Closed mouth: the "opening" is only the seam between the lips. Painting it (its own darkness survives as
        # texture) avoids an unpainted grey line through the lipstick.
        return a_outer.astype(np.float32)
    return (a_outer * a_inner * (inner == 0)).astype(np.float32)


def _gaussian_spot(shape, center: np.ndarray, rx: float, ry: float) -> np.ndarray:
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return np.exp(-(((xs - center[0]) / max(rx, 0.5)) ** 2 + ((ys - center[1]) / max(ry, 0.5)) ** 2) * 2.0)


def _masked_blur(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Blur that only averages lip pixels, so skin and teeth never leak into the lip's shading."""
    num = cv2.GaussianBlur(img * weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return num / np.maximum(den, 1e-4)


def _highlights(shape, local_pts: np.ndarray, prof: FinishProfile, L: np.ndarray, lip_w: np.ndarray,
                mw: float) -> np.ndarray:
    """L* to add per pixel: where light sits on the lipstick for this finish."""
    total = np.zeros(shape, dtype=np.float32)
    if prof.lower_sheen > 0:
        # Fullest part of the lower lip, a little below the mouth line: a smooth horizontal band
        top, bottom = local_pts[LOWER_INNER_MID], local_pts[LOWER_OUTER_MID]
        lip_h = float(np.linalg.norm(bottom - top))
        center = top + (bottom - top) * 0.42
        total += _gaussian_spot(shape, center, mw * prof.lower_sheen_rx, max(lip_h * prof.lower_sheen_ry, 0.8)) * prof.lower_sheen
    if prof.peak_sheen > 0:
        for outer_i, inner_i in UPPER_PEAKS:
            o, i = local_pts[outer_i], local_pts[inner_i]
            center = o + (i - o) * 0.45
            h = float(np.linalg.norm(i - o))
            total += _gaussian_spot(shape, center, mw * 0.05, max(h * 0.16, 0.8)) * prof.peak_sheen
    if prof.photo_specular > 0:
        # Only the photo's few brightest lip spots, smoothed first so skin pores and noise never become white specks
        lip_L = cv2.GaussianBlur(L, (0, 0), sigmaX=max(1.0, mw * 0.012))
        vals = lip_L[lip_w > 0.5]
        if vals.size:
            p93, p995 = np.percentile(vals, [93, 99.5])
            spec = np.clip((lip_L - p93) / max(float(p995 - p93), 1.0), 0.0, 1.0)
            total += spec * spec * (3.0 - 2.0 * spec) * prof.photo_specular
    sigma = max(0.6, mw * prof.sheen_blur)
    return cv2.GaussianBlur(total, (0, 0), sigmaX=sigma, sigmaY=sigma) * lip_w


def shade_lab(hex_color: str) -> np.ndarray:
    """CIELAB of a product shade as it looks under the venue's light (beauty_tuning.SCENE_LIGHT_GAINS)."""
    lin = _to_linear(hex_to_rgb(hex_color).astype(np.float32) / 255.0) * np.asarray(SCENE_LIGHT_GAINS, np.float32)
    rgb = _to_srgb(np.clip(lin, 0.0, 1.0)).astype(np.float32)
    return cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]


def _recolor(roi_rgb: np.ndarray, lip_w: np.ndarray, local_pts: np.ndarray, target_lab: np.ndarray,
             finish: LipstickFinish, mw: float) -> np.ndarray:
    """Recoloured copy of the crop (float32 RGB 0-1). `lip_w` (0-1) weights which pixels are lip."""
    prof = FINISHES[finish]
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)  # float input: L 0-100, a/b about -127..127
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    t_L, t_a, t_b = (float(v) for v in target_lab)

    hard = (lip_w > 0.5).astype(np.float32)
    sigma = max(1.5, mw * SHADING_SIGMA_FRAC)
    L_low = _masked_blur(L, hard, sigma)
    lip_L = float(np.median(L_low[hard > 0]))
    texture = L - L_low  # fine detail: creases, pores
    shading = L_low - lip_L  # broad 3D form: the lit and shadowed sides of each lip
    texture = np.where(texture > 0, texture * prof.texture_highlight_gain, texture)

    # Base colour: the shade, mixed back toward the lip's natural tone by k. Pigment never fully hides the lip, and a
    # shade lighter than the lip hides it least (a nude on pigmented lips reads deeper and warmer than its swatch).
    lip_a, lip_b = float(np.median(a[hard > 0])), float(np.median(b[hard > 0]))
    k = NATURAL_TONE_BASE + NATURAL_TONE_LIGHT * float(np.clip((t_L - lip_L) / NATURAL_TONE_LIGHT_SPAN, 0.0, 1.0))
    base_L, base_a, base_b = t_L + (lip_L - t_L) * k, t_a + (lip_a - t_a) * k, t_b + (lip_b - t_b) * k

    # Lightness = base + the lip's own shading and texture (that is what keeps it from looking like a sticker);
    # a*/b* = base chroma + a little of the lip's own fine colour variation
    new_L = base_L + prof.lightness_offset + shading * prof.shading_gain + texture * prof.texture_gain
    new_a = base_a * prof.chroma_gain + (a - _masked_blur(a, hard, sigma)) * CHROMA_DETAIL
    new_b = base_b * prof.chroma_gain + (b - _masked_blur(b, hard, sigma)) * CHROMA_DETAIL

    light = _highlights(L.shape, local_pts, prof, L, lip_w, mw)
    if light.any():
        new_L = new_L + light
        keep = 1.0 - np.clip(light / 30.0, 0.0, 1.0) * prof.highlight_desat
        new_a, new_b = new_a * keep, new_b * keep

    out_lab = np.dstack([np.clip(new_L, 0.0, 100.0), new_a, new_b]).astype(np.float32)
    return np.clip(cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.maximum(c, 0.0031308) ** (1 / 2.4) - 0.055)


def composite(base: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Float sRGB (0-1) `layer` covering `alpha` (H, W; 0-1) of `base`, mixed in linear light.

    A sheer product is pigment covering part of the surface, so what mixes is the light each part reflects (linear
    values), not the gamma-encoded pixel values. Averaging sRGB values instead always lands darker (the sRGB decoding
    curve is convex), which made every partial setting look heavier than its opacity. alpha 0 returns `base` exactly.
    """
    a = alpha[..., None]
    mixed = _to_srgb(_to_linear(base) * (1.0 - a) + _to_linear(layer) * a)
    return np.where(a > 0.0, mixed, base)


def apply_lipstick(rgb: np.ndarray, pts: np.ndarray, hex_color: str, finish: LipstickFinish = LipstickFinish.SATIN,
                   intensity: float = 0.75) -> np.ndarray:
    """Applies the shade to the lips described by `pts` (478x2 face-mesh pixel coords). Returns a new RGB uint8 image.

    `intensity` is the slider position (0-1); beauty_tuning.opacity maps it to how much of the lip the layer covers.
    """
    if len(pts) < MESH_POINTS:
        raise ValueError("Lip contours need a full face mesh")
    out = np.array(rgb[..., :3], dtype=np.uint8, copy=True)
    layer_opacity = opacity(intensity, LIP_OPACITY_MIN, LIP_OPACITY_MAX)
    if layer_opacity <= 0.0:
        return out
    h, w = out.shape[:2]
    target = shade_lab(hex_color)
    mw = mouth_width(pts)

    lips = pts[LIPS_OUTER]
    pad = mw * ROI_PAD_FRAC + EDGE_FEATHER_RANGE[1] * 3
    x0, y0 = np.floor(np.maximum(lips.min(axis=0) - pad, 0)).astype(int)
    x1, y1 = np.ceil(np.minimum(lips.max(axis=0) + pad, [w, h])).astype(int)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return out

    local = (pts - np.array([x0, y0], dtype=np.float32)).astype(np.float32)
    roi = out[y0:y1, x0:x1].astype(np.float32) / 255.0
    mask = build_lip_mask(roi.shape[:2], local)
    if not (mask > 0.5).any():
        return out

    colored = _recolor(roi, mask, local, target, finish, mw)
    # The only place opacity is applied: mask (edges, corners) x the slider's curve, once
    out[y0:y1, x0:x1] = np.round(composite(roi, colored, mask * layer_opacity) * 255.0).astype(np.uint8)
    return out


_detector = None
_detector_lock = threading.Lock()


def _face_model_path() -> Optional[str]:
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(getattr(sys, "_MEIPASS", ""), "assets", "face_landmarker.task"),
        os.path.join(here, "..", "..", "assets", "face_landmarker.task"),
        os.path.join("assets", "face_landmarker.task"),
    ]
    return next((os.path.abspath(p) for p in candidates if os.path.exists(p)), None)


def _get_detector():
    """This module's own FaceLandmarker: requests run in a worker thread, so they never share the clothing pipeline's
    instance (which it calls from the event loop) and never contend with it."""
    global _detector
    if _detector is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        path = _face_model_path()
        if not path:
            raise RuntimeError("assets/face_landmarker.task not found")
        _detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            num_faces=1,
            min_face_detection_confidence=0.4,
        ))
        print(f"[BEAUTY] FaceLandmarker initialized from: {path}")
    return _detector


def detect_lip_points(rgb: np.ndarray) -> Optional[np.ndarray]:
    from app.services.face_preservation import _detect_mesh
    with _detector_lock:
        detector = _get_detector()
        pts = _detect_mesh(detector, rgb)
        if pts is None:  # full-length capture: the face can be too small for the detector in the whole frame
            pts = _detect_mesh(detector, rgb[: rgb.shape[0] // 2])
    return pts


def locate_face(rgb: np.ndarray, require_lips: bool = True) -> np.ndarray:
    """Face-mesh points of a face large enough for makeup; raises LipsNotFound (visitor-safe message) otherwise."""
    pts = detect_lip_points(rgb)
    if pts is None or len(pts) < MESH_POINTS:
        raise LipsNotFound("We couldn't find your face. Please face the camera and try again.")
    if not require_lips:
        if float(np.linalg.norm(pts[234] - pts[454])) < MIN_FACE_WIDTH_PX:
            raise LipsNotFound("Your face is too far from the camera for a powder preview. Take a close-up instead.")
        return pts
    if mouth_width(pts) < MIN_MOUTH_WIDTH_PX:
        raise LipsNotFound("Your face is too far from the camera for a lipstick preview. Take a close-up instead.")
    return pts


def try_on_lipstick(image: Image.Image, hex_color: str, finish: LipstickFinish = LipstickFinish.SATIN,
                    intensity: float = 0.75) -> LipstickResult:
    rgb = np.ascontiguousarray(np.array(image.convert("RGB")))
    pts = locate_face(rgb)
    h, w = rgb.shape[:2]
    result = apply_lipstick(rgb, pts, hex_color, finish, intensity)
    return LipstickResult(image=result, face_box=face_box(pts, w, h), mouth_width_px=mouth_width(pts))
