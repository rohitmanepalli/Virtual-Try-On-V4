"""
Size Chart & Recommendation Engine.

Maps visitor body-profile proportions (normalized ratios from BodyProfileService)
to standard garment sizes (XS – XXL) and computes per-size garment-to-body
fabric offset values that drive the Gemini prompt conditioning.

Because BodyProfile stores *normalized ratios* (e.g. shoulder_width_ratio ≈ 0.28)
rather than absolute centimetre measurements, size classification uses
proportion-band matching and body-type heuristics rather than a direct cm lookup.
"""

from dataclasses import dataclass, field
from math import inf, isinf
from typing import Any, Dict, List, Optional, Tuple

from app.models import BodyProfile


# ─── Standard Sizes ───────────────────────────────────────────────────────────

ALL_SIZES: List[str] = ["XS", "S", "M", "L", "XL", "XXL"]

SIZE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(ALL_SIZES)}


# ─── Proportion Band Definitions ──────────────────────────────────────────────
#
# Each band maps a body-proportion range to a size.
# The ranges overlap slightly at boundaries; the engine picks the best match
# by scoring distance-from-center across chest, waist, hip, and shoulder width.
#
# chest_proportion / waist_proportion / hip_proportion are Ramanujan-ellipse
# circumference ratios normalized against hip_circ (so hip is always 1.0).
# shoulder_width_ratio is the raw bi-acromial width in frame-normalized coords.
#
# These defaults approximate unisex casual / semi-tailored sizing.
# Garment-category overrides are applied on top when available.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _PropBand:
    """Proportion-band definition for a single size."""
    chest_lo: float
    chest_hi: float
    waist_lo: float
    waist_hi: float
    shoulder_lo: float
    shoulder_hi: float


# Proportion bands keyed by size label
_DEFAULT_BANDS: Dict[str, _PropBand] = {
    #        chest_lo  chest_hi  waist_lo  waist_hi  shoulder_lo  shoulder_hi
    "XS":  _PropBand(0.00,     1.10,     0.00,     0.76,     0.00,        0.225),
    "S":   _PropBand(1.05,     1.22,     0.72,     0.83,     0.210,       0.255),
    "M":   _PropBand(1.15,     1.35,     0.80,     0.90,     0.240,       0.285),
    "L":   _PropBand(1.25,     1.45,     0.87,     0.97,     0.275,       0.325),
    "XL":  _PropBand(1.35,     1.55,     0.94,     1.05,     0.315,       0.365),
    "XXL": _PropBand(1.45,     inf,      1.02,     inf,      0.355,       inf),
}


def get_size_prop_band(size_label: str) -> Optional[_PropBand]:
    """Returns the proportion band definition for a given size label."""
    if not size_label:
        return None
    return _DEFAULT_BANDS.get(size_label.upper().strip())

# Category-specific shoulder bias: tailored outerwear (blazers, suits) runs
# wider in the shoulders for the same size vs. soft garments (t-shirts, dresses).
_CATEGORY_SHOULDER_BIAS: Dict[str, float] = {
    "blazer": 0.015,
    "suit_jacket": 0.015,
    "suit": 0.015,
    "coat": 0.02,
    "jacket": 0.015,
    "shirt": 0.0,
    "t_shirt": -0.005,
    "dress": -0.01,
    "kurti": 0.01,
    "kurta": 0.01,
    "sherwani": 0.015,
    "kurta_dhoti": 0.01,
    "salwar_suit": 0.01,
    "kurta_pajama": 0.01,
    "lehenga": 0.01,
    "royal_sherwani": 0.015,
    "saree": 0.01,
    "mundu_shirt": 0.0,
    "tunic": 0.01,
    "hoodie": 0.01,
}


# ─── Size-to-Offset Mapping ──────────────────────────────────────────────────
#
# Defines the garment-to-body fabric offset (in mm) for each size step relative
# to the visitor's recommended size.  These offsets directly feed the Gemini
# prompt conditioning and must match the values documented in the plan.
#
# steps_from_recommended → (offset_lo_mm, offset_hi_mm, fit_quality, does_not_fit)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SizeOffset:
    """Fit of one selected size on this visitor, relative to their recommended size."""
    steps_from_recommended: int
    offset_lo_mm: float
    offset_hi_mm: float
    fit_quality: str       # one of FIT_LABELS
    does_not_fit: bool
    fit_label: str = ""
    recommended_size: Optional[str] = None
    selected_size: Optional[str] = None
    # Smaller sizes only: size steps by which each body dimension exceeds the selected size's design maximum
    body_check: Dict[str, float] = field(default_factory=dict)
    # The dimension that limits this size (largest excess over its stretch allowance); where strain shows first
    limiting_dimension: Optional[str] = None
    wont_fit_reason: Optional[str] = None


# Fit categories by offset: +1 relaxed, +2 clearly loose, +3 or more very oversized, 0 correct, -1 snug,
# -2 tight with visible strain. "does_not_fit" is decided by the body-versus-size check or an offset of -3 or lower.
FIT_LABELS: Dict[str, str] = {
    "oversized": "Very oversized",
    "loose": "Loose fit",
    "relaxed": "Relaxed fit",
    "perfect": "Correct fit",
    "snug": "Snug fit",
    "tight": "Tight fit",
    "does_not_fit": "Won't fit",
}
LOOSE_FITS = ("relaxed", "loose", "oversized")
TIGHT_FITS = ("snug", "tight", "does_not_fit")

_OFFSET_TABLE: Dict[int, Tuple[float, float, str]] = {
    # Standard garments: (lo_mm, hi_mm, quality); negative = garment smaller than the body, fabric under tension
     3: (60.0, 80.0, "oversized"),
     2: (45.0, 60.0, "loose"),
     1: (30.0, 45.0, "relaxed"),
     0: (15.0, 25.0, "perfect"),
    -1: ( 0.0, 10.0, "snug"),
    -2: (-10.0, 0.0, "tight"),
}

_KURTI_OFFSET_TABLE: Dict[int, Tuple[float, float, str]] = {
    # Kurtis have an airy, loose drape with higher baseline ease (20-35mm at step 0)
     3: (70.0, 90.0, "oversized"),
     2: (50.0, 70.0, "loose"),
     1: (35.0, 50.0, "relaxed"),
     0: (20.0, 35.0, "perfect"),
    -1: ( 5.0, 15.0, "snug"),
    -2: (-5.0,  5.0, "tight"),
}

# Straight-cut tunic tops, alone or in a set: sized by the top (shoulder and chest led) with the kurti ease above
_KURTA_CATEGORIES = ("kurti", "kurta", "kurta_dhoti", "kurta_pajama", "salwar_suit", "tunic")
# Sets sized by a fitted top (blouse or shirt) with the same weighting but standard ease; their draped parts are unsized
_FITTED_TOP_SETS = ("lehenga", "saree", "mundu_shirt")

WONT_FIT_OFFSET_MM = (-25.0, -10.0)


def _clamp_steps(steps: int) -> int:
    """Clamp extreme offsets to the defined range (+3 and above share the very-oversized entry)."""
    return max(-2, min(3, steps))


# ─── Body-versus-size check (smaller sizes) ───────────────────────────────────
#
# Each size's design maximum for a dimension is its band's upper bound. A body dimension beyond it is expressed in
# size steps (one step = the gap to the next size's maximum), so the same threshold works for chest, waist and
# shoulder. The allowance is how many steps past its maximum a garment can still be forced closed: the garment's own
# ease plus fabric stretch. Structured tailoring cannot stretch (and padded shoulders not at all); knits and soft
# drapes stretch the most. Provisional: set from garment-ease rules of thumb, not measured on real garments.
_SIZE_CHECK_DIMS = ("chest", "waist", "shoulder")

STRETCH_ALLOWANCE_STEPS: Dict[str, Dict[str, float]] = {
    "rigid_tailored": {"chest": 1.0, "waist": 1.25, "shoulder": 0.75},
    "semi_structured": {"chest": 1.25, "waist": 1.25, "shoulder": 1.0},
    "soft_drape": {"chest": 1.5, "waist": 1.5, "shoulder": 1.25},
}

# Used when no garment analysis is available (e.g. /api/recommend-size, which only knows the category)
_CATEGORY_STRUCTURE: Dict[str, str] = {
    "blazer": "rigid_tailored", "suit_jacket": "rigid_tailored", "suit": "rigid_tailored",
    "coat": "rigid_tailored", "jacket": "rigid_tailored", "sherwani": "rigid_tailored",
    "royal_sherwani": "rigid_tailored",
    "t_shirt": "soft_drape", "hoodie": "soft_drape", "kurti": "soft_drape", "tunic": "soft_drape",
    "salwar_suit": "soft_drape",
}

_BODY_ATTR = {"chest": "chest_proportion", "waist": "waist_proportion", "shoulder": "shoulder_width_ratio"}


def _structure_for(garment_category: Optional[str], structure_level: Optional[str]) -> str:
    if structure_level in STRETCH_ALLOWANCE_STEPS:
        return structure_level
    return _CATEGORY_STRUCTURE.get((garment_category or "").lower().strip(), "semi_structured")


def _design_max(size: str, dim: str) -> float:
    """Largest body value the size is designed for; XXL (open-ended band) extends one XL-sized step past XL."""
    hi = getattr(_DEFAULT_BANDS[size], f"{dim}_hi")
    if isinf(hi):
        xl, l = getattr(_DEFAULT_BANDS["XL"], f"{dim}_hi"), getattr(_DEFAULT_BANDS["L"], f"{dim}_hi")
        return xl + (xl - l)
    return hi


def _size_step(size: str, dim: str) -> float:
    idx = SIZE_INDEX[size]
    if idx + 1 < len(ALL_SIZES):
        return _design_max(ALL_SIZES[idx + 1], dim) - _design_max(size, dim)
    return _design_max(size, dim) - _design_max(ALL_SIZES[idx - 1], dim)


def body_excess_steps(body_profile: BodyProfile, size: str) -> Dict[str, float]:
    """Per dimension: size steps by which the body exceeds the size's design maximum (<= 0: within the size)."""
    size = size.upper().strip()
    return {
        dim: round((getattr(body_profile, _BODY_ATTR[dim]) - _design_max(size, dim)) / _size_step(size, dim), 2)
        for dim in _SIZE_CHECK_DIMS
    }


# ─── Public API ───────────────────────────────────────────────────────────────

@dataclass
class SizeRecommendationResult:
    """Full recommendation output returned to the caller."""
    recommended_size: str
    available_sizes: List[str]
    size_confidence: float
    fit_notes: str


def _match_dim(val: float, dim_name: str) -> str:
    """
    Finds which size band [lo, hi] the measurement falls into by walking
    ALL_SIZES in ascending order (XS -> XXL).
    If below all bands, returns XS.
    If above all bands, returns XXL.
    """
    xs_lo = getattr(_DEFAULT_BANDS["XS"], f"{dim_name}_lo")
    if val < xs_lo:
        return "XS"

    for size in ALL_SIZES:
        band = _DEFAULT_BANDS[size]
        lo = getattr(band, f"{dim_name}_lo")
        hi = getattr(band, f"{dim_name}_hi")
        if lo <= val <= hi:
            return size

    return "XXL"


def compute_size_from_measurements(
    chest: float,
    waist: float,
    shoulder: float,
    garment_category: Optional[str] = None
) -> str:
    """
    Deterministic range matching for body measurements:
    1. Independent size votes for chest, waist, shoulder walking XS->XXL.
    2. Category-specific tie-breaking:
       - Blazers/jackets/coats: shoulder vote is primary; if chest differs by >1 step, average toward chest.
       - Sherwanis: same as blazers. The structured jacket is sized at the shoulders and chest; the draped dhoti is
         the same at every size, so the waist does not vote. A layered royal sherwani is sized by its outer layer.
       - Dresses: waist vote is primary; if chest differs by >1 step, average toward chest.
       - Shirts/t-shirts: median of the three votes.
    """
    if chest is None or waist is None or shoulder is None:
        raise ValueError("Missing required body measurement data.")

    chest_size = _match_dim(chest, "chest")
    waist_size = _match_dim(waist, "waist")
    shoulder_size = _match_dim(shoulder, "shoulder")

    cat = (garment_category or "").lower().strip()

    if cat in ("blazer", "suit_jacket", "suit", "coat", "jacket", "sherwani", "royal_sherwani"):
        sh_idx = SIZE_INDEX[shoulder_size]
        ch_idx = SIZE_INDEX[chest_size]
        if abs(sh_idx - ch_idx) > 1:
            avg_idx = int(round((sh_idx + ch_idx) / 2.0))
            return ALL_SIZES[avg_idx]
        return shoulder_size
    elif cat in ("dress",):
        w_idx = SIZE_INDEX[waist_size]
        ch_idx = SIZE_INDEX[chest_size]
        if abs(w_idx - ch_idx) > 1:
            avg_idx = int(round((w_idx + ch_idx) / 2.0))
            return ALL_SIZES[avg_idx]
        return waist_size
    elif cat in _KURTA_CATEGORIES + _FITTED_TOP_SETS:
        # Straight-cut Kurtis and Kurtas are driven by chest and shoulder ease, with waist de-emphasized:
        # 0.40 * shoulder + 0.40 * chest + 0.20 * waist. A kurta with dhoti is sized by its kurta: the dhoti is
        # wrapped to fit and never changes the size; a salwar suit by its kameez (the salwar and dupatta are unsized);
        # a kurta with pants by its kurta (the pants then show the same fit category, see fit_drape); a lehenga by its
        # fitted blouse, a saree by its blouse and a mundu set by its shirt (same weighting, standard ease; the skirt,
        # saree drape, mundu, dupatta and shawl are unsized).
        sh_idx = SIZE_INDEX[shoulder_size]
        ch_idx = SIZE_INDEX[chest_size]
        w_idx = SIZE_INDEX[waist_size]
        weighted_val = 0.40 * sh_idx + 0.40 * ch_idx + 0.20 * w_idx
        idx = int(round(weighted_val))
        idx = max(0, min(len(ALL_SIZES) - 1, idx))
        return ALL_SIZES[idx]
    else:  # shirts, t-shirts, unisex casual
        votes = sorted([SIZE_INDEX[chest_size], SIZE_INDEX[waist_size], SIZE_INDEX[shoulder_size]])
        return ALL_SIZES[votes[1]]


def recommend_size(
    body_profile: BodyProfile,
    garment_category: Optional[str] = None
) -> SizeRecommendationResult:
    """
    Matches a visitor's captured body profile against the garment size chart
    using deterministic dimension votes and category rules.
    """
    if not body_profile:
        raise ValueError("body_profile is required for size recommendation.")

    chest = body_profile.chest_proportion
    waist = body_profile.waist_proportion
    shoulder = body_profile.shoulder_width_ratio

    best_size = compute_size_from_measurements(
        chest=chest,
        waist=waist,
        shoulder=shoulder,
        garment_category=garment_category,
    )

    sh_vote = _match_dim(shoulder, "shoulder")
    ch_vote = _match_dim(chest, "chest")
    w_vote = _match_dim(waist, "waist")

    # Fit notes
    notes_parts: List[str] = []
    if SIZE_INDEX[sh_vote] > SIZE_INDEX[best_size]:
        notes_parts.append("Broad shoulders — may prefer sizing up for shoulder room")
    elif SIZE_INDEX[sh_vote] < SIZE_INDEX[best_size]:
        notes_parts.append("Narrow shoulders — garment may sit wide at shoulders")

    if SIZE_INDEX[ch_vote] > SIZE_INDEX[best_size]:
        notes_parts.append("Fuller chest — check chest fit at this size")

    if SIZE_INDEX[w_vote] > SIZE_INDEX[best_size]:
        notes_parts.append("Waist runs large — consider sizing up for comfort")

    body_type = body_profile.body_type or ""
    if "Athletic" in body_type or "V-Shape" in body_type:
        notes_parts.append("Athletic build — shoulders may need a half-size up in tailored garments")
    elif "Hourglass" in body_type or "Curved" in body_type:
        notes_parts.append("Hourglass build — fitted cuts will accentuate waist")
    elif "Pear" in body_type or "Triangle" in body_type:
        notes_parts.append("Pear build — hip room is the limiting dimension")

    fit_notes = "; ".join(notes_parts) if notes_parts else f"Good fit in {best_size} across all dimensions"

    matching_votes = sum(1 for v in (sh_vote, ch_vote, w_vote) if v == best_size)
    if matching_votes == 3:
        confidence = 1.0
    elif matching_votes == 2:
        confidence = 0.85
    else:
        confidence = 0.70

    print(
        f"[SIZE REC] Recommended: {best_size} (confidence {confidence:.0%}) | "
        f"Votes: [chest={ch_vote}, waist={w_vote}, shoulder={sh_vote}] | "
        f"Category: {garment_category or 'generic'} | Body: {body_type}"
    )

    return SizeRecommendationResult(
        recommended_size=best_size,
        available_sizes=list(ALL_SIZES),
        size_confidence=confidence,
        fit_notes=fit_notes,
    )


def get_size_offset(
    recommended_size: str,
    selected_size: str,
    garment_category: Optional[str] = None,
    body_profile: Optional[BodyProfile] = None,
    structure_level: Optional[str] = None,
) -> SizeOffset:
    """
    Fit category of the selected size on this visitor. The offset (selected minus recommended, on XS..XXL) sets the
    category; for smaller sizes the visitor's chest, waist and shoulder are checked against the selected size's chart
    maximum plus the garment's stretch allowance, which decides "won't fit" versus snug / tight. So at the same offset
    a lean visitor can be "tight" where a broader one "won't fit". Without a body profile only an offset of -3 or
    lower is "won't fit".
    """
    rec = recommended_size.upper().strip()
    sel = selected_size.upper().strip()
    rec_idx = SIZE_INDEX.get(rec, 2)   # default M = index 2
    sel_idx = SIZE_INDEX.get(sel, rec_idx)
    raw_steps = sel_idx - rec_idx   # positive = larger, negative = smaller

    cat = (garment_category or "").lower().strip()
    table = _KURTI_OFFSET_TABLE if cat in _KURTA_CATEGORIES else _OFFSET_TABLE
    lo, hi, quality = table[_clamp_steps(raw_steps)]

    body_check: Dict[str, float] = {}
    limiting: Optional[str] = None
    reason: Optional[str] = None
    if raw_steps < 0 and body_profile is not None and sel in SIZE_INDEX:
        body_check = body_excess_steps(body_profile, sel)
        allowance = STRETCH_ALLOWANCE_STEPS[_structure_for(cat, structure_level)]
        limiting = max(body_check, key=lambda d: body_check[d] - allowance[d])
        over = body_check[limiting] - allowance[limiting]
        if over > 0:
            quality = "does_not_fit"
            reason = (f"{limiting} exceeds size {sel} by {body_check[limiting]:.2f} size steps "
                      f"(stretch allowance {allowance[limiting]:.2f})")
    if raw_steps <= -3:
        quality = "does_not_fit"
        reason = reason or f"{-raw_steps} sizes below the recommended size {rec}"
    if quality == "does_not_fit":
        lo, hi = WONT_FIT_OFFSET_MM

    return SizeOffset(
        steps_from_recommended=raw_steps,
        offset_lo_mm=lo,
        offset_hi_mm=hi,
        fit_quality=quality,
        does_not_fit=quality == "does_not_fit",
        fit_label=FIT_LABELS[quality],
        recommended_size=rec,
        selected_size=sel,
        body_check=body_check,
        limiting_dimension=limiting,
        wont_fit_reason=reason,
    )


def size_fit_map(
    body_profile: Optional[BodyProfile],
    recommended_size: str,
    garment_category: Optional[str] = None,
    structure_level: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fit category and label of every size for this visitor, so the selector can label any size before it renders."""
    fits = {}
    for size in ALL_SIZES:
        o = get_size_offset(recommended_size, size, garment_category, body_profile, structure_level)
        fits[size] = {"fit_quality": o.fit_quality, "fit_label": o.fit_label, "size_offset": o.steps_from_recommended}
    return fits
