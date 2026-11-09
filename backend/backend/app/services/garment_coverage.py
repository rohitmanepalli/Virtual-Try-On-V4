"""
Garment coverage: the selected garment alone decides what is covered.

There is no inpaint mask. The model redraws the whole frame from the capture and the garment, so a long sleeve that
shows below a shorter one is drawn by the model, not left over from the capture. Coverage is set in the prompt and
checked on every render:

1. read_capture() runs once per request. Where the visitor's own sleeves end and what is on their wrists come from the
   appearance analysis (a webcam lights the face and the forearms too differently for a pixel test to tell a sleeve
   from skin). The pixels supply the cheek skin colour and the colours of the visitor's own sleeve on each forearm.
2. coverage_section() tells the model where the selected garment's sleeves and hem end, and what shows beyond them.
3. check_render() looks at each forearm of a render where the visitor's own sleeve was and the selected garment's is
   not. The old sleeve's colours there are a remnant and fail the render. Where the selected garment's sleeve reaches
   but the visitor's arm was bare, bare skin is a gap and fails the render.

Every checked stretch stops short of the wrist, where a watch, bracelet or kada sits. Anything unmeasurable (no face,
no wrist landmark, a sleeve colour too close to skin) is "not checked" and never fails a render.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from app.models import ClothAnalysis

SLEEVE_LENGTHS = ("sleeveless", "short", "elbow", "three_quarter", "full")
TOP_LENGTHS = ("cropped", "waist", "hip", "thigh", "knee", "below_knee")
# What the visitor wears below the waist in the capture (appearance analysis): decides the lower body of a top-only render
OWN_BOTTOMS = ("separate", "one_piece", "not_visible")
# Used only when the garment analysis did not say: catalog profiles and the heuristic fallback
_BOTTOM_CATEGORIES = ("dress", "frock", "gown", "jumpsuit", "saree", "lehenga", "salwar_suit", "kurta_pajama",
                      "kurta_dhoti", "sherwani", "royal_sherwani", "mundu_shirt", "kurti", "kurta", "suit")
# Top-only garments that get jeans when the visitor's one-piece is replaced; everything else gets plain trousers
_CASUAL_TOPS = ("t_shirt", "tshirt", "polo_shirt", "polo", "hoodie", "sweatshirt", "tunic", "top", "tank_top", "crop_top")

# Where each sleeve length ends, along the forearm: 0 = elbow, 1 = wrist, negative = above the elbow
SLEEVE_END = {"sleeveless": -1.0, "short": -0.45, "elbow": 0.0, "three_quarter": 0.5, "full": 1.0}
SLEEVE_END_MARGIN = 0.20  # a render's sleeve may end this far past its nominal end before the stretch is judged
WRIST_GUARD = 0.88  # judged stretches end here: past it sit the watch, bracelet or kada
MIN_ZONE = 0.15  # shortest stretch worth judging
COVER_START = 0.30  # a sleeve to the wrist is judged from here
CAPTURE_SLEEVE_ZONE = (0.15, 0.80)  # where the visitor's own full sleeve is sampled
UPPER_ARM_ZONE = (0.35, 0.80)  # along shoulder -> elbow: under any sleeve except sleeveless
HAND_ZONE = (0.25, 0.75)  # along wrist -> index knuckle: the back of the hand, past any watch or bracelet
STRIP_HALF_WIDTH = 0.07  # half-width of the sampled strip around the bone line, as a fraction of the limb length
HAND_HALF_WIDTH = 0.25
MIN_STRIP_PIXELS = 40

# Colour distance in CIELAB with lightness down-weighted: relighting changes lightness far more than hue
L_WEIGHT = 0.5
SKIN_MATCH = 12.0  # a pixel within this of the skin reference is skin
SLEEVE_MATCH = 20.0  # a render pixel within this of an old sleeve colour (and nearer to it than to skin) is a remnant
SLEEVE_DISTINCT = 18.0  # sleeve colours nearer than this to skin cannot be told apart from bare arm: not judged
REMNANT_MAX_SHARE = 0.30  # old-sleeve pixels allowed in a stretch that must be bare
GAP_MAX_SHARE = 0.40  # skin pixels allowed in a stretch that must be covered

CHEEK_POINTS = (50, 205, 280, 425)  # MediaPipe face mesh: mid-cheek, below spectacle rims and above a beard line
DETECT_MAX_SIDE = 1600

_SLEEVE_TEXT = {
    "sleeveless": "The selected garment is sleeveless: both arms are bare from the shoulder to the hand.",
    "short": "The selected garment's sleeves are short and end on the upper arm, above the elbow: both arms are bare "
             "from the sleeve's edge to the hand, elbows and forearms included.",
    "elbow": "The selected garment's sleeves end at the elbow: both forearms are bare from the elbow to the hand.",
    "three_quarter": "The selected garment's sleeves end at mid-forearm: the lower half of both forearms, the wrists "
                     "and the hands are bare.",
    "full": "The selected garment's sleeves run to the wrist: they cover both arms from the shoulder to the wrist.",
}
_OWN_SLEEVE_WORDS = {
    "sleeveless": "no sleeves", "short": "short sleeves", "elbow": "elbow-length sleeves",
    "three_quarter": "three-quarter sleeves", "full": "long sleeves to the wrist",
}
_HEM_TEXT = {
    "waist": "the waist", "hip": "the hips", "thigh": "mid-thigh", "knee": "the knee", "below_knee": "below the knee",
}


def normalize_length(value: Any, allowed: Sequence[str]) -> Optional[str]:
    """An analysis answer mapped onto one of `allowed` (None when it is not one of them)."""
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"half": "short", "half_sleeve": "short", "short_sleeve": "short", "short_sleeves": "short",
               "3/4": "three_quarter", "three_quarters": "three_quarter", "long": "full", "long_sleeve": "full",
               "long_sleeves": "full", "full_sleeve": "full", "full_sleeves": "full", "rolled_up": "elbow",
               "none": "sleeveless", "midriff": "cropped", "crop": "cropped", "calf": "below_knee",
               "ankle": "below_knee", "floor": "below_knee"}
    text = aliases.get(text, text)
    return text if text in allowed else None


@dataclass
class CaptureCoverage:
    own_sleeves: Optional[str] = None  # where the visitor's own sleeves end (appearance analysis)
    wrist_accessories: Optional[str] = None  # as the appearance analysis described them; None = none seen
    own_bottom: Optional[str] = None  # one of OWN_BOTTOMS (appearance analysis); None = unknown
    skin_lab: Optional[List[float]] = None  # cheeks, CIELAB
    skin_hex: Optional[str] = None  # cheeks, for the prompt
    hand_lab: Optional[List[float]] = None  # backs of the hands, CIELAB: lit like the forearms, unlike the face
    sleeve_colors: Dict[str, List[List[float]]] = field(default_factory=dict)  # per forearm, CIELAB palette

    def sleeves_longer_than(self, garment_sleeve: Optional[str]) -> Optional[bool]:
        if self.own_sleeves not in SLEEVE_END or garment_sleeve not in SLEEVE_END:
            return None
        return SLEEVE_END[self.own_sleeves] > SLEEVE_END[garment_sleeve]

    def sleeves_shorter_than(self, garment_sleeve: Optional[str]) -> Optional[bool]:
        if self.own_sleeves not in SLEEVE_END or garment_sleeve not in SLEEVE_END:
            return None
        return SLEEVE_END[self.own_sleeves] < SLEEVE_END[garment_sleeve]

    def summary(self) -> str:
        palette = ", ".join(f"{side} {len(c)} colour(s)" for side, c in self.sleeve_colors.items()) or "none"
        return (f"own sleeves {self.own_sleeves or 'unknown'} | own bottom {self.own_bottom or 'unknown'} | wrists: {self.wrist_accessories or 'none'} | "
                f"skin {self.skin_hex or 'n/a'} (hands {'measured' if self.hand_lab else 'n/a'}) | own sleeve palette: {palette}")


def _lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB pixels (..., 3) -> float CIELAB (N, 3): L 0-100, a/b about -128..127."""
    px = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _dist(lab: np.ndarray, ref: Sequence[float]) -> np.ndarray:
    d = np.asarray(lab, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
    return np.sqrt((L_WEIGHT * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _hex(lab: Sequence[float]) -> str:
    rgb = cv2.cvtColor(np.array([[lab]], dtype=np.float32), cv2.COLOR_LAB2RGB)[0, 0]
    r, g, b = (int(round(float(np.clip(c, 0.0, 1.0)) * 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _downscale(rgb: np.ndarray) -> np.ndarray:
    """A <= DETECT_MAX_SIDE copy for 4K renders: colour statistics do not need full resolution."""
    h, w = rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / float(max(h, w)))
    if scale >= 1.0:
        return rgb
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def skin_reference(rgb: np.ndarray, face_pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Median CIELAB skin colour on the cheeks. The darker 40% (beard, stubble, shadow) and the brightest 5% (specular
    highlights) of the sampled pixels are dropped.
    """
    if face_pts is None or len(face_pts) < max(CHEEK_POINTS) + 1:
        return None
    face_w = float(np.linalg.norm(face_pts[234] - face_pts[454]))
    radius = max(2.0, 0.05 * face_w)
    h, w = rgb.shape[:2]
    sel = np.zeros((h, w), dtype=bool)
    for idx in CHEEK_POINTS:
        cx, cy = face_pts[idx]
        x1, x2 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
        y1, y2 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        sel[y1:y2, x1:x2] |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    if sel.sum() < 12:
        return None
    lab = _lab(rgb[sel])
    order = np.argsort(lab[:, 0])
    lo = int(len(order) * 0.40)
    kept = lab[order[lo:max(lo + 1, int(len(order) * 0.95))]]
    return np.median(kept, axis=0)


def strip_pixels(rgb: np.ndarray, lm: Any, start: str, end: str, t0: float, t1: float,
                 half_width: float = STRIP_HALF_WIDTH) -> Optional[np.ndarray]:
    """
    Pixels in a thin strip along the limb from keypoint `start` to `end`, between fractions t0 and t1 of its length,
    restricted to the person's segmentation mask. None when either keypoint is missing (MediaPipe drops wrists it
    cannot see clearly) or the landmarks came from the contour fallback.
    """
    if lm is None or not getattr(lm, "visibility", None):
        return None
    kp = lm.keypoints or {}
    a, b = kp.get(start), kp.get(end)
    if not a or not b:
        return None
    h, w = rgb.shape[:2]
    p0 = np.array([a[0] * w, a[1] * h], dtype=np.float32)
    p1 = np.array([b[0] * w, b[1] * h], dtype=np.float32)
    length = float(np.linalg.norm(p1 - p0))
    if length < 12.0:
        return None
    axis = (p1 - p0) / length
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    half = half_width * length
    ends = np.array([p0 + axis * t0 * length, p0 + axis * t1 * length])
    x1, y1 = np.floor(ends.min(0) - half).astype(int)
    x2, y2 = np.ceil(ends.max(0) + half).astype(int) + 1
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return None
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    rel_x, rel_y = xx - p0[0], yy - p0[1]
    s = (rel_x * axis[0] + rel_y * axis[1]) / length
    n = np.abs(rel_x * normal[0] + rel_y * normal[1])
    sel = (s >= t0) & (s <= t1) & (n <= half)
    mask = getattr(lm, "segmentation_mask", None)
    if mask is not None:
        m = np.squeeze(mask)
        if m.ndim == 2:
            mh, mw = m.shape
            my = np.clip((yy / h * mh).astype(int), 0, mh - 1)
            mx = np.clip((xx / w * mw).astype(int), 0, mw - 1)
            sel &= m[my, mx] > 0.5
    if sel.sum() < MIN_STRIP_PIXELS:
        return None
    return rgb[y1:y2, x1:x2][sel]


def sleeve_palette(lab: np.ndarray) -> List[List[float]]:
    """Up to three dominant colours of a sleeve (a check or a stripe has several), each over 15% of its pixels."""
    if len(lab) < MIN_STRIP_PIXELS:
        return []
    k = 3
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(lab.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / float(len(labels))
    return [centers[i].round(2).tolist() for i in range(k) if shares[i] >= 0.15]


def _own_sleeve_zone(own_sleeves: Optional[str]) -> Optional[Tuple[float, float]]:
    """Where the visitor's own sleeve can be sampled on the capture's forearm (None if it does not reach the forearm)."""
    if own_sleeves not in SLEEVE_END:
        return None
    t0, t1 = CAPTURE_SLEEVE_ZONE[0], min(CAPTURE_SLEEVE_ZONE[1], SLEEVE_END[own_sleeves] - 0.05)
    return (t0, t1) if t1 - t0 >= MIN_ZONE else None


def hand_reference(rgb: np.ndarray, lm: Any) -> Optional[np.ndarray]:
    """
    Median CIELAB skin colour on the backs of both hands, which are lit like the forearms (a face under a ceiling light
    or a studio key light is much brighter than the arms). The darkest 20% (shadow between fingers) and the brightest
    5% are dropped. None when neither hand is measurable.
    """
    parts = [strip_pixels(rgb, lm, f"{side}_wrist", f"{side}_index", *HAND_ZONE, half_width=HAND_HALF_WIDTH)
             for side in ("left", "right")]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    lab = _lab(np.concatenate(parts))
    order = np.argsort(lab[:, 0])
    lo = int(len(order) * 0.20)
    return np.median(lab[order[lo:max(lo + 1, int(len(order) * 0.95))]], axis=0)


def read_capture(portrait: Image.Image, details: Optional[Dict[str, Optional[str]]] = None) -> CaptureCoverage:
    """
    Garment-coverage facts about the capture portrait (the image the model receives). `details` are this capture's
    appearance-analysis answers (own_sleeves, wrist_accessories).
    """
    from app.services.body_profile import body_profile_service
    from app.services.face_preservation import detect_face_points

    details = details or {}
    coverage = CaptureCoverage(own_sleeves=details.get("own_sleeves"), wrist_accessories=details.get("wrist_accessories"),
                               own_bottom=details.get("own_bottom"))
    rgb = np.array(portrait.convert("RGB"))
    skin = skin_reference(rgb, detect_face_points(rgb))
    if skin is not None:
        coverage.skin_lab, coverage.skin_hex = skin.round(2).tolist(), _hex(skin)
    lm = body_profile_service.extract_single_frame(portrait)
    hand = hand_reference(rgb, lm)
    if hand is not None:
        coverage.hand_lab = hand.round(2).tolist()
    zone = _own_sleeve_zone(coverage.own_sleeves)
    if zone is not None:
        for side in ("left", "right"):
            px = strip_pixels(rgb, lm, f"{side}_elbow", f"{side}_wrist", *zone)
            if px is not None:
                coverage.sleeve_colors[side] = sleeve_palette(_lab(px))
    return coverage


def _relight(color: Sequence[float], skin_from: Optional[Sequence[float]], skin_to: Optional[Sequence[float]]) -> np.ndarray:
    """A capture colour as it would look under the render's lighting, using the skin's change as the measure (the
    capture colour unchanged when no skin pair was measured)."""
    if skin_from is None or skin_to is None:
        return np.asarray(color, dtype=np.float32)
    lightness = color[0] * skin_to[0] / max(float(skin_from[0]), 1.0)
    return np.array([min(100.0, lightness), color[1] + skin_to[1] - skin_from[1], color[2] + skin_to[2] - skin_from[2]],
                    dtype=np.float32)


def _check_bare(rgb, lm, side, zone, capture, relight_pair, render_skin) -> Dict[str, Any]:
    """The visitor's own sleeve must be gone from this stretch: count pixels nearer its colours than skin."""
    palette = [_relight(c, *relight_pair) for c in capture.sleeve_colors.get(side, [])]
    palette = [c for c in palette if float(_dist(c, render_skin)) >= SLEEVE_DISTINCT]
    if not palette:
        return {"checked": False, "reason": "own sleeve colour not measured or too close to skin"}
    px = strip_pixels(rgb, lm, f"{side}_elbow", f"{side}_wrist", *zone)
    if px is None:
        return {"checked": False, "reason": "render forearm not measurable"}
    lab = _lab(px)
    d_skin = _dist(lab, render_skin)
    d_sleeve = np.min(np.stack([_dist(lab, c) for c in palette]), axis=0)
    remnant = float(np.mean((d_sleeve < d_skin) & (d_sleeve < SLEEVE_MATCH)))
    skin = float(np.mean(d_skin < SKIN_MATCH))
    passed = not (remnant > REMNANT_MAX_SHARE and remnant > skin)
    return {"checked": True, "kind": "bare", "zone": [round(z, 2) for z in zone], "remnant_share": round(remnant, 3),
            "skin_share": round(skin, 3), "passed": passed}


def _check_covered(rgb, lm, side, zone, render_skin) -> Dict[str, Any]:
    """The selected garment's sleeve must cover this stretch: count pixels nearer skin than the new sleeve's colour."""
    upper = strip_pixels(rgb, lm, f"{side}_shoulder", f"{side}_elbow", *UPPER_ARM_ZONE)
    px = strip_pixels(rgb, lm, f"{side}_elbow", f"{side}_wrist", *zone)
    if upper is None or px is None:
        return {"checked": False, "reason": "render arm not measurable"}
    new_sleeve = np.median(_lab(upper), axis=0)
    if float(_dist(new_sleeve, render_skin)) < SLEEVE_DISTINCT:
        return {"checked": False, "reason": "new sleeve colour too close to skin"}
    lab = _lab(px)
    d_skin = _dist(lab, render_skin)
    gap = float(np.mean((d_skin < SKIN_MATCH) & (d_skin < _dist(lab, new_sleeve))))
    return {"checked": True, "kind": "covered", "zone": [round(z, 2) for z in zone], "skin_share": round(gap, 3),
            "passed": gap <= GAP_MAX_SHARE}


def check_arms(rgb: np.ndarray, lm: Any, capture: Optional[CaptureCoverage], sleeve_length: Optional[str],
               render_skin: Optional[np.ndarray],
               relight_pair: Tuple[Optional[Sequence[float]], Optional[Sequence[float]]] = (None, None)) -> Dict[str, Any]:
    """
    Per-forearm coverage verdict on a render (pixel work only; see check_render). render_skin: the render's skin
    reference (its hands when visible, else its cheeks). relight_pair: the same body part's skin in the capture and in
    the render, used to carry the old sleeve's colours into the render's lighting ((None, None) = no relighting).
    """
    result: Dict[str, Any] = {"checked": False, "passed": None, "problem": None, "sides": None,
                              "sleeve_length": sleeve_length, "arms": {}}
    if sleeve_length not in SLEEVE_END:
        result["reason"] = "garment sleeve length unknown"
        return result
    if capture is None or capture.own_sleeves not in SLEEVE_END:
        result["reason"] = "visitor's own sleeve length unknown"
        return result
    if render_skin is None:
        result["reason"] = "no skin reference"
        return result
    own_end, new_end = SLEEVE_END[capture.own_sleeves], SLEEVE_END[sleeve_length]
    if own_end > new_end:
        # The visitor's own sleeve reached further than the selected garment's: that stretch must be bare
        zone, kind = (max(MIN_ZONE, new_end + SLEEVE_END_MARGIN), min(WRIST_GUARD, own_end)), "remnant"
    elif own_end < new_end:
        # The selected garment's sleeve reaches further than the visitor's own: that stretch must be covered
        zone, kind = (max(COVER_START, own_end + 0.10), min(WRIST_GUARD, new_end) - 0.07), "gap"
    else:
        result["reason"] = "same sleeve length as the visitor's own"
        return result
    if zone[1] - zone[0] < MIN_ZONE:
        result["reason"] = "stretch to judge too short"
        return result
    failed = []
    for side in ("left", "right"):
        arm = (_check_bare(rgb, lm, side, zone, capture, relight_pair, render_skin) if kind == "remnant"
               else _check_covered(rgb, lm, side, zone, render_skin))
        result["arms"][side] = arm
        if arm.get("checked") and not arm["passed"]:
            failed.append(side)
    result["checked"] = any(a.get("checked") for a in result["arms"].values())
    if result["checked"]:
        result["passed"] = not failed
        if failed:
            result["problem"], result["sides"] = kind, " and ".join(failed)
    return result


def check_render(capture: Optional[CaptureCoverage], render: Image.Image, render_lm: Any,
                 sleeve_length: Optional[str]) -> Dict[str, Any]:
    """
    Does the render's coverage match the selected garment? passed True/False when at least one forearm was judged,
    None when nothing was measurable (never a failure). render_lm: the render's pose landmarks (normalised, so they
    apply to the downscaled copy unchanged). Skin is judged against the render's hands (lit like its forearms) when
    visible, else its cheeks; the old sleeve's colours are relit hand to hand or cheek to cheek, whichever pair both
    images show.
    """
    from app.services.face_preservation import detect_face_points

    rgb = np.zeros((1, 1, 3), np.uint8)
    render_skin, relight_pair, source = None, (None, None), None
    if capture is not None and sleeve_length in SLEEVE_END and capture.sleeves_longer_than(sleeve_length) is not None:
        rgb = _downscale(np.array(render.convert("RGB")))
        hand = hand_reference(rgb, render_lm)
        face = skin_reference(rgb, detect_face_points(rgb)) if hand is None or capture.hand_lab is None else None
        render_skin, source = (hand, "hands") if hand is not None else (face, "face" if face is not None else None)
        if hand is not None and capture.hand_lab is not None:
            relight_pair = (capture.hand_lab, hand)
        elif face is not None and capture.skin_lab is not None:
            relight_pair = (capture.skin_lab, face)
    result = check_arms(rgb, render_lm, capture, sleeve_length, render_skin, relight_pair)
    result["skin_reference"] = source
    return result


def log_line(result: Optional[Dict[str, Any]], attempt_no: int) -> str:
    if not result:
        return f"[COVERAGE CHECK] Attempt {attempt_no}: not run"
    arms = []
    for side, a in result.get("arms", {}).items():
        if a.get("checked"):
            share = (f"old sleeve {a['remnant_share']:.0%} skin {a['skin_share']:.0%}" if a["kind"] == "bare"
                     else f"skin {a['skin_share']:.0%}")
            arms.append(f"{side} {share} {'ok' if a['passed'] else 'FAIL'}")
        else:
            arms.append(f"{side} not checked ({a.get('reason')})")
    verdict = {True: "PASS", False: f"FAIL: {result.get('problem')} on {result.get('sides')} forearm", None: "not checked"}
    return (f"[COVERAGE CHECK] Attempt {attempt_no} | garment sleeves {result.get('sleeve_length')} | "
            f"{'; '.join(arms) or result.get('reason', '')} | {verdict[result.get('passed')]}")


def garment_has_bottom(cloth: Optional[ClothAnalysis]) -> bool:
    """Does the selected garment dress the legs itself (outfit set, dress, saree)? The analysis answer, else category."""
    if cloth is None:
        return False
    if getattr(cloth, "includes_bottom", None) is not None:
        return cloth.includes_bottom
    return (cloth.category or "").lower() in _BOTTOM_CATEGORIES


def lower_body_text(cloth: Optional[ClothAnalysis], capture: Optional[CaptureCoverage]) -> str:
    """
    What the lower body wears, decided here and not left to the model: the outfit's own bottom; else the visitor's own
    separate bottom kept as it is; else (their one-piece is replaced by a top, or it is out of frame) a plain bottom
    chosen for the top's style. The legs are never bare.
    """
    if garment_has_bottom(cloth):
        return ("- LOWER BODY: the selected outfit includes its own lower garment (Image 2), which replaces the person's "
                "lower garment completely.\n")
    category = (getattr(cloth, "category", None) or "").lower()
    bottom = ("plain mid-blue denim jeans" if category in _CASUAL_TOPS else
              "plain, well-fitted trousers in a neutral colour that suits the top (black, charcoal, navy, beige or off-white)")
    own = capture.own_bottom if capture is not None else None
    if own == "separate":
        return ("- LOWER BODY — KEEP THE PERSON'S OWN: the selected garment is a top only, and in Image 1 the person "
                "wears a separate lower garment (trousers, jeans, a skirt). Keep it exactly as it is: the same type, "
                "colour, fabric, fit and length, continued naturally where Image 1 is cropped. Only the top changes; "
                "the lower garment never disappears and the legs are never bare.\n")
    if own == "one_piece":
        return ("- LOWER BODY — NEW PLAIN BOTTOM: in Image 1 the person wears a one-piece outfit (a dress, frock, gown "
                "or jumpsuit), which the selected top replaces entirely: no skirt, hem or fabric of it remains below the "
                f"new top. Dress the legs in {bottom}, full length to the ankles, with no print or decoration. The legs "
                "are never bare.\n")
    return ("- LOWER BODY: the selected garment is a top only. If the person wears a separate lower garment in Image 1, "
            "keep it exactly as it is; if they wear a one-piece outfit or their lower body is not visible, dress the legs "
            f"in {bottom}, full length to the ankles. The legs are never bare.\n")


def coverage_section(cloth: Optional[ClothAnalysis], capture: Optional[CaptureCoverage]) -> str:
    """The prompt section that says where the selected garment ends and what shows beyond it."""
    sleeve = getattr(cloth, "sleeve_length", None)
    top = getattr(cloth, "top_length", None)
    wrists = capture.wrist_accessories if capture is not None else None
    if sleeve not in SLEEVE_END and top not in TOP_LENGTHS and not wrists:
        return "\n[GARMENT COVERAGE]\n" + lower_body_text(cloth, capture)
    lines =["\n[GARMENT COVERAGE — THE SELECTED GARMENT REPLACES THE PERSON'S OWN TOP]"]
    own = capture.own_sleeves if capture is not None else None
    if sleeve in SLEEVE_END:
        lines.append(f"- {_SLEEVE_TEXT[sleeve]}")
        longer = capture.sleeves_longer_than(sleeve) if capture is not None else None
        shorter = capture.sleeves_shorter_than(sleeve) if capture is not None else None
        if SLEEVE_END[sleeve] < 1.0:
            if longer:
                lines.append(
                    f"- In Image 1 the person wears {_OWN_SLEEVE_WORDS[own]}, longer than the selected garment's. They "
                    "belong to the person's own top, which the selected garment replaces: remove them completely. Below "
                    "the new sleeve's edge there is no cuff, no fabric edge, no fold and none of the old sleeve's "
                    "colour, pattern or texture: only the bare arm.")
            elif longer is None:
                lines.append(
                    "- If the person wears longer sleeves in Image 1, remove them completely: below the new sleeve's "
                    "edge there is no cuff, fabric edge or trace of the old sleeve, only the bare arm.")
            tone = f" (measured on the cheeks in Image 1: about {capture.skin_hex})" if capture is not None and capture.skin_hex else ""
            lines.append(
                f"- The bare arms are the person's own skin, exactly the tone and undertone of their face and hands in "
                f"Image 1{tone}, with the same natural texture. Do not lighten, darken or recolour it.")
        if shorter:
            lines.append(
                f"- In Image 1 the person wears {_OWN_SLEEVE_WORDS[own]}, shorter than the selected garment's. The new "
                "sleeves cover that bare skin completely down to their own end: no skin shows through gaps, slits or a "
                "sleeve that stops short.")
        elif shorter is None and SLEEVE_END[sleeve] > SLEEVE_END["sleeveless"]:
            lines.append("- If the person's arms are barer in Image 1, the new sleeves still cover the arm fully down to "
                         "their own end, with no skin showing through gaps.")
    if wrists:
        lines.append(
            f"- WRIST ACCESSORIES IN IMAGE 1: {wrists}. Every one of them stays exactly as it is: the same wrist, "
            "position, colour, material and size. Changing the sleeve around a wrist changes only the fabric: never "
            "drop an accessory, replace it, or turn it into a different one.")
    else:
        lines.append(
            "- WRIST ACCESSORIES: any watch, bracelet, bangle, kada or thread on a wrist in Image 1 stays exactly as it "
            "is, on the same wrist. Changing the sleeve around it changes only the fabric, never the accessory.")
    if top == "cropped":
        lines.append(
            "- The selected garment's top is cropped and ends above the waist. The person's own top must not show "
            "below it: between its hem and the outfit's lower garment there is only bare midriff skin (or the outfit's "
            "own drape over it), the same skin tone as the arms and face.")
    elif top in _HEM_TEXT:
        lines.append(
            f"- The selected garment's top ends at {_HEM_TEXT[top]}. If the person's own top in Image 1 is longer, it "
            "must not show below the new hem: below it is only the outfit's own lower garment when Image 2 includes "
            "one, otherwise the lower garment described under LOWER BODY.")
    lines.append(lower_body_text(cloth, capture).rstrip("\n"))
    lines.append("- A necklace or chain at the neck in Image 1 stays exactly as it is, resting on the new neckline or on "
                 "the skin where the old collar was.")
    return "\n".join(lines) + "\n"


def coverage_escalation_text(result: Dict[str, Any], sleeve_length: Optional[str]) -> str:
    """Directive for the retry after a render failed the coverage check."""
    sides = result.get("sides") or "one"
    if result.get("problem") == "gap":
        return (
            "\n[COVERAGE RETRY — A PREVIOUS ATTEMPT WAS REJECTED]\n"
            f"- It showed bare skin on the {sides} forearm where the selected garment's sleeve must cover the arm. Draw "
            "the full sleeve down to its own end on both arms, with no skin showing through.\n"
        )
    where = {"sleeveless": "shoulder", "short": "short sleeve", "elbow": "elbow-length sleeve",
             "three_quarter": "three-quarter sleeve"}.get(sleeve_length or "", "sleeve")
    return (
        "\n[COVERAGE RETRY — A PREVIOUS ATTEMPT WAS REJECTED]\n"
        f"- It showed the person's own longer sleeve below the selected garment's {where} on the {sides} forearm. That "
        "sleeve must be gone: from the new sleeve's edge to the hand the arm is bare skin on both sides, with no cuff, "
        "fabric edge or old-sleeve colour. Keep every watch, bracelet or kada on the wrists exactly as in Image 1.\n"
    )
