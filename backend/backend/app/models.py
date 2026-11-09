from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class FitState(str, Enum):
    FITTED = "fitted"
    STANDARD = "standard"
    RELAXED = "relaxed"


class BodyProfile(BaseModel):
    height_ratio: float = Field(..., description="Estimated head-to-ankle proportion of frame")
    shoulder_width_ratio: float = Field(..., description="Bi-acromial shoulder breadth relative to frame")
    torso_aspect_ratio: float = Field(..., description="Torso length relative to shoulder width")
    shoulder_to_hip_ratio: float = Field(..., description="Transverse shoulder width / hip width")
    chest_proportion: float = Field(..., description="Estimated chest proportion")
    waist_proportion: float = Field(..., description="Estimated waist proportion")
    hip_proportion: float = Field(..., description="Estimated hip proportion")
    body_type: str = Field(..., description="Categorized body type (e.g. Athletic / V-Shape, Hourglass / Curved, etc.)")
    confidence_score: float = Field(1.0, description="Mean pose confidence across the capture frames")
    frame_count_sampled: int = Field(1, description="Number of capture frames used for the median measurements")
    # Body volume signal for the generation prompt and silhouette verifier only — never read by size matching
    torso_fullness: Optional[float] = Field(
        None, description="Mean midsection silhouette width (arms excluded) / shoulder-joint width; None when unmeasurable"
    )
    torso_shape: Optional[Literal["lean", "average", "full", "very_full"]] = Field(
        None, description="Band of torso_fullness"
    )
    keypoints_summary: Optional[Dict[str, Any]] = None


MeasurementField = Literal["height", "shoulder_width", "chest_width", "waist_width"]


class BodyMeasurement(BaseModel):
    value_cm: float
    # Scale comes from an assumed standing distance, so every value is currently "estimated"
    confidence: Literal["measured", "estimated"] = "estimated"
    source: str = Field(..., description="silhouette | landmarks | shoulder_ratio")


class BodyMeasurements(BaseModel):
    """Informational cm estimates for display only. Kept separate from BodyProfile so they never reach sizing or try-on."""
    height: Optional[BodyMeasurement] = None
    shoulder_width: Optional[BodyMeasurement] = None
    chest_width: Optional[BodyMeasurement] = None
    waist_width: Optional[BodyMeasurement] = None
    assumed_distance_m: float


class MeasurementCorrectionRequest(BaseModel):
    session_id: str
    field: MeasurementField
    displayed_cm: float = Field(..., description="Value shown to the visitor before editing (may already be rescaled)")
    corrected_cm: float = Field(..., gt=0, lt=300)


class ProfileCaptureRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded front-facing capture photo")
    hold_frames: List[str] = Field(
        default_factory=list,
        max_length=2,
        description="Up to 2 extra base64 frames grabbed while the visitor holds still, used only for median jitter suppression",
    )
    session_id: Optional[str] = None


class FaceCheckRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded capture photo, exactly as later sent to /api/tryon")


class PrepareCaptureRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded accepted capture photo, exactly as later sent to /api/tryon")


class ClothAnalysis(BaseModel):
    category: str = Field(..., description="Detected garment category: blazer, suit_jacket, coat, shirt, t_shirt, dress, etc.")
    garment_name: str = Field(..., description="Concise descriptive title of garment")
    fabric_type: str = Field(..., description="Fabric material description and texture")
    structure_level: str = Field("semi_structured", description="Structure level: rigid_tailored, semi_structured, or soft_drape")
    is_outerwear: bool = Field(False, description="Whether the garment is an outerwear piece layered over undergarments")
    sleeve_behavior: str = Field(..., description="Explicit sleeve and arm drape behavior guidelines")
    tryon_guidelines: str = Field(..., description="Detailed virtual try-on rendering guidelines for this specific garment")
    # Where the garment ends: the prompt's coverage section and the render's coverage check (see garment_coverage)
    sleeve_length: Optional[Literal["sleeveless", "short", "elbow", "three_quarter", "full"]] = Field(
        None, description="Where the sleeves end; None when unknown"
    )
    top_length: Optional[Literal["cropped", "waist", "hip", "thigh", "knee", "below_knee"]] = Field(
        None, description="Where the top's hem ends (cropped = above the waist, midriff showing); None when unknown"
    )
    includes_bottom: Optional[bool] = Field(
        None, description="True when the garment covers the legs itself (outfit set with its own lower garment, dress, "
                          "saree, gown, jumpsuit); False for a top only; None when unknown (decided by category)"
    )


class SizeRecommendation(BaseModel):
    recommended_size: str = Field(..., description="Best-fit standard size for the visitor (e.g. M)")
    available_sizes: List[str] = Field(default_factory=list, description="All available sizes for this garment")
    size_confidence: float = Field(1.0, description="Confidence of the size recommendation (0–1)")
    fit_notes: str = Field("", description="Human-readable fit notes for the visitor")
    size_fits: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per size: fit_quality, fit_label and size_offset for this visitor (body-checked for smaller sizes)",
    )


class TryOnRequest(BaseModel):
    user_image: str
    dress_image: str
    fit_state: Optional[str] = "standard"
    selected_size: Optional[str] = None
    body_profile: Optional[BodyProfile] = None
    cloth_analysis: Optional[ClothAnalysis] = None
    session_id: Optional[str] = None
    last_good_render: Optional[str] = None
    retry_strict: Optional[bool] = False
    garment_id: Optional[str] = None  # Catalog id of the outfit, for the visitor records


class VisitorDetailsRequest(BaseModel):
    """Every field is a short limited string, so a request cannot carry an image into the visitor records."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., max_length=80)
    consent: bool = False
    consent_version: str = Field("", max_length=40)
    name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=24)
    gender: Optional[Literal["female", "male", "non_binary", "prefer_not_to_say"]] = None


class VisitorDetailsResponse(BaseModel):
    saved: bool


class SilhouetteVerificationResult(BaseModel):
    score: float
    passed: bool
    threshold: float
    details: Optional[Dict[str, Any]] = None


class TryOnResponse(BaseModel):
    success: bool
    result_image: Optional[str] = None
    qr_code_base64: Optional[str] = None
    fit_state: Optional[str] = "standard"
    selected_size: Optional[str] = None
    size_recommendation: Optional[SizeRecommendation] = None
    size_fit_quality: Optional[str] = None
    size_offset: Optional[int] = Field(None, description="Selected size index minus recommended size index")
    fit_label: Optional[str] = Field(None, description="Short fit label for the selected size, e.g. 'Relaxed fit'")
    # "Won't fit" is its own state, not a failed render: the message is always shown. result_image is a render of the
    # garment visibly too small on the visitor's body, or None when no render depicted it believably
    # (illustration_only: the client shows the message with an illustration built from fit_check).
    wont_fit: bool = False
    illustration_only: bool = False
    fit_check: Optional[Dict[str, Any]] = Field(
        None, description="Won't-fit details: per-dimension size steps past the size, limiting dimension, reason"
    )
    # False when the served render could not be verified to differ from the other sizes (see logs); None when not checked
    fit_verified: Optional[bool] = None
    # False when the served render still shows the visitor's own sleeve past the garment's sleeve, or bare skin under
    # it (see logs); None when not checked (unknown sleeve lengths, nothing measurable)
    coverage_verified: Optional[bool] = None
    cloth_analysis: Optional[ClothAnalysis] = None
    silhouette_score: Optional[float] = None
    silhouette_passed: Optional[bool] = None
    face_similarity: Optional[float] = None
    fallback_used: bool = False
    verification_failed: bool = False
    needs_visitor_realignment: bool = False
    message: Optional[str] = None


class PurgeSessionRequest(BaseModel):
    session_id: str


class PurgeSessionResponse(BaseModel):
    success: bool
    session_id: str
    message: str


# --- Beauty Studio: lipstick and compact powder (independent of the clothing pipeline above) ---

class LipstickFinish(str, Enum):
    MATTE = "matte"
    SATIN = "satin"
    GLOSS = "gloss"


class CompactPowderFinish(str, Enum):
    MATTE = "matte"
    NATURAL_VELVET = "natural_velvet"
    LUMINOUS = "luminous"


class LipstickShade(BaseModel):
    """Catalog entry served by GET /api/beauty/shades."""
    id: str
    name: str
    category: str = Field(..., description="Shade family, e.g. Classic Reds, Nudes & Browns")
    hex_color: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    finish_default: LipstickFinish = LipstickFinish.SATIN


class PowderShade(BaseModel):
    """Catalog entry served by GET /api/beauty/powders."""
    id: str
    name: str
    hex_tint: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$", description="Swatch colour")
    tinted: bool = Field(True, description="False for translucent powder: no colour is added to the skin")
    finish_default: CompactPowderFinish = CompactPowderFinish.NATURAL_VELVET


class LipstickConfig(BaseModel):
    shade_id: str
    shade_name: str
    hex_color: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    finish: LipstickFinish = LipstickFinish.SATIN
    intensity: float = Field(0.85, ge=0.0, le=1.0)


class CompactPowderConfig(BaseModel):
    shade_id: str
    shade_name: str
    hex_tint: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    finish: CompactPowderFinish = CompactPowderFinish.NATURAL_VELVET
    coverage: Literal["sheer", "medium"] = "medium"


class BeautyTryOnRequest(BaseModel):
    image: str  # Base64 portrait
    session_id: Optional[str] = None
    lipstick: Optional[LipstickConfig] = None
    compact_powder: Optional[CompactPowderConfig] = None
    # Only "Save look" asks for the download copy + QR (and the studio portrait). Live previews never write to
    # temp_images, so they can't push a visitor's clothing download out of the MAX_TEMP_IMAGES window.
    include_qr: bool = False


class BeautyTryOnResponse(BaseModel):
    success: bool
    mode: str  # "lipstick_only" | "powder_only" | "full_makeover"
    result_image: str  # Base64 output ("" when success is False)
    qr_code_base64: Optional[str] = None
    active_look: Dict[str, Any]
    message: Optional[str] = None
    # Face bounding box in the result, normalized 0-1 (x, y, w, h), so the client can zoom into a full-length capture
    face_box: Optional[Dict[str, float]] = None
    # True when the saved look is the studio portrait (Gemini lighting/backdrop + the visitor's real face + makeup)
    studio_render: bool = False
