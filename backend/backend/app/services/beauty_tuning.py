"""
Beauty Studio look tuning: every value that changes how lipstick and compact powder look, in one place.

Opacity curve (lipstick slider and powder coverage):

    opacity(t) = MIN + (MAX - MIN) * t ** OPACITY_EXPONENT    for t in (0, 1]
    opacity(0) = 0                                             the photo comes back untouched

t is what the app sends: lipstick slider position / 100, or POWDER_COVERAGE_T for a powder coverage. The exponent > 1
spends more of the slider's travel on the sheer end, where the eye notices small changes most, and MIN makes the
first step a visible tint (a tinted lip balm) instead of nothing. The opacity is the share of the lip/skin covered by
the product layer; the layer itself is mixed in linear light (lipstick_service.composite).
"""
from dataclasses import dataclass
from typing import Dict

from app.models import CompactPowderFinish, LipstickFinish

# --- Opacity curve -----------------------------------------------------------------------------------------------------
OPACITY_EXPONENT = 1.6
LIP_OPACITY_MIN, LIP_OPACITY_MAX = 0.08, 0.85  # barely-there tint .. full coverage that still shows lip texture
POWDER_OPACITY_MIN, POWDER_OPACITY_MAX = 0.08, 0.65  # enough for shades to read apart; never a heavy layer
POWDER_COVERAGE_T = {"sheer": 0.55, "medium": 1.0}  # coverage chip -> position on the powder curve


def opacity(t: float, lo: float, hi: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return 0.0 if t <= 0.0 else lo + (hi - lo) * t ** OPACITY_EXPONENT


# --- Venue light (calibration) -------------------------------------------------------------------------------------
# Linear-RGB gains of the venue's light relative to daylight, applied to every lipstick and powder shade before it goes
# on the photo, so a shade under warm indoor light warms up like the skin around it instead of looking pasted on.
# (1, 1, 1) = daylight or a white-balanced camera. To calibrate: photograph a plain white card at the kiosk; the gains
# are the card's linear RGB divided by its luminance (0.2126 R + 0.7152 G + 0.0722 B). ~(1.33, 0.96, 0.46) is ~3000 K.
SCENE_LIGHT_GAINS = (1.0, 1.0, 1.0)


# --- Lip mask (shares of the mouth width; px ranges clamp them) ----------------------------------------------------
EDGE_INSET_FRAC = 0.006  # outer edge pulled inside the landmark contour (landmarks sit a hair outside the vermilion)
EDGE_FEATHER_FRAC = 0.012  # outer edge falloff width
EDGE_FEATHER_RANGE = (1.0, 3.0)  # px
INNER_FEATHER_FRAC = 0.006  # mouth-opening falloff width
INNER_FEATHER_RANGE = (0.6, 2.0)  # px
CORNER_FADE_RADIUS_FRAC = 0.10  # the lipstick thins out over this distance into each mouth corner...
CORNER_FADE_DEPTH = 0.35  # ...down to (1 - this) of full strength at the corner itself
CLOSED_GAP_FRAC = 0.06  # lips parted less than this (inner gap / mouth width) are painted through the seam
MASK_SUPERSAMPLE_MOUTH_PX = 160  # narrower mouths get their mask drawn at up to 4x, so small lips keep smooth edges

# --- Lip colour ----------------------------------------------------------------------------------------------------
SHADING_SIGMA_FRAC = 0.045  # scale separating the lip's broad 3D shading from its fine texture (creases)
CHROMA_DETAIL = 0.15  # share of the lip's own fine a*/b* variation kept under the shade
# Natural lip tone showing through the pigment, as a share of the way from the shade back to the lip's own colour.
# Always NATURAL_TONE_BASE; up to + NATURAL_TONE_LIGHT more for a shade lighter than the lip (a nude on pigmented
# lips reads deeper and warmer than its swatch), reached when the shade is NATURAL_TONE_LIGHT_SPAN L* lighter.
NATURAL_TONE_BASE = 0.12
NATURAL_TONE_LIGHT = 0.35
NATURAL_TONE_LIGHT_SPAN = 25.0


@dataclass(frozen=True)
class FinishProfile:
    shading_gain: float  # share of the lip's broad 3D shading kept
    texture_gain: float  # share of fine texture (creases) kept
    texture_highlight_gain: float  # extra scale on the bright side of the texture (matte kills tiny shine)
    lightness_offset: float  # L* added to the shade (matte reads deeper, gloss a touch lighter)
    chroma_gain: float  # saturation of the shade
    lower_sheen: float  # L* of the lower-lip highlight
    lower_sheen_rx: float  # its half-width, as a share of the mouth width
    lower_sheen_ry: float  # its half-height, as a share of the lower lip's height
    sheen_blur: float  # softness of every highlight, as a share of the mouth width (bigger = more diffuse)
    peak_sheen: float  # L* of the highlights on the upper-lip peaks
    photo_specular: float  # L* added where the photo's own light already hits the lip
    highlight_desat: float  # chroma removed under highlights (white light on a wet surface)


FINISHES: Dict[LipstickFinish, FinishProfile] = {
    LipstickFinish.MATTE: FinishProfile(0.45, 0.70, 0.35, -2.0, 1.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    LipstickFinish.SATIN: FinishProfile(0.65, 0.85, 0.90, 0.0, 1.00, 10.0, 0.26, 0.40, 0.030, 0.0, 0.0, 0.20),
    LipstickFinish.GLOSS: FinishProfile(0.70, 0.55, 1.00, 1.5, 0.97, 30.0, 0.17, 0.16, 0.010, 14.0, 8.0, 0.70),
}

# --- Compact powder (shares of the face width) ---------------------------------------------------------------------
OVAL_INSET_FRAC = 0.035  # forehead and temples: stay this far inside the face oval (off the hairline)
OVAL_FEATHER_FRAC = 0.04
JAW_EXTEND_FRAC = 0.06  # jaw: the powder runs this far past the jawline into the neck...
JAW_FEATHER_FRAC = 0.12  # ...fading out over this width, so there is no edge between face and neck
FEATURE_MARGIN_FRAC = 0.025  # clearance around eyes/brows/lips
TEXTURE_SIGMA_FRAC = 0.02  # fine texture (pores) below this scale is always kept
EVEN_SIGMA_FRAC = 0.07  # blotchiness is evened toward this broader average
# The shade shows as undertone and depth, on the low-frequency layer only: pores and texture stay untouched, so a
# stronger tint does not turn plastic (that comes from evening/smoothing, which stays as it was)
TINT_CHROMA = 0.45  # share of the way the skin's a*/b* moves toward the powder shade
TINT_LIGHTNESS = 0.25  # same for L*...
TINT_LIGHTNESS_MAX = 7.0  # ...capped at this many L* either way: a shade far from the skin is a visible shift, never a mask


@dataclass(frozen=True)
class PowderFinishProfile:
    even_strength: float  # share of the way blotchiness moves toward the broad average
    shine_margin: float  # L* above the skin reference where shine control starts
    shine_strength: float  # share of that excess removed
    texture_keep: float  # pores stay, just a touch softer, like finely milled powder
    glow: float  # L* lifted on the face's high points (luminous only)


POWDER_FINISHES: Dict[CompactPowderFinish, PowderFinishProfile] = {
    # Gentle on purpose: strong evening, full shine removal or softened pores is what reads as a "powdery doll" face
    CompactPowderFinish.MATTE: PowderFinishProfile(0.30, 4.0, 0.55, 0.97, 0.0),
    CompactPowderFinish.NATURAL_VELVET: PowderFinishProfile(0.25, 6.0, 0.40, 0.98, 0.0),
    CompactPowderFinish.LUMINOUS: PowderFinishProfile(0.22, 9.0, 0.15, 0.99, 3.5),
}
GLOW_SPAN = 8.0  # L* above the skin reference over which the luminous lift ramps up
