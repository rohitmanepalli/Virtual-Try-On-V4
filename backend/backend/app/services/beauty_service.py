"""
Beauty Studio processing: one entry point for lipstick only, compact powder only, or both (a full makeover).

Engines: lipstick_service (lips) and powder_service (facial skin), both local and deterministic, sharing one face-mesh
detection. Powder goes on first, lipstick on top, as in real application. The paid studio portrait for a saved look
lives in beauty_studio_service and is orchestrated by the router.
"""
from typing import Any, Dict, Optional

import numpy as np

from app.models import BeautyTryOnRequest, CompactPowderConfig, LipstickConfig
from app.services import lipstick_service as ls
from app.services.beauty_tuning import POWDER_COVERAGE_T
from app.services.powder_service import apply_powder

MODE_LIPSTICK = "lipstick_only"
MODE_POWDER = "powder_only"
MODE_FULL = "full_makeover"

TRANSLUCENT_IDS = {"translucent"}  # colourless powders: shine control and soft focus, no tint


def look_mode(req: BeautyTryOnRequest) -> str:
    if req.lipstick and req.compact_powder:
        return MODE_FULL
    if req.compact_powder:
        return MODE_POWDER
    return MODE_LIPSTICK


def active_look(req: BeautyTryOnRequest) -> Dict[str, Any]:
    return {
        "lipstick": req.lipstick.model_dump(mode="json") if req.lipstick else None,
        "compact_powder": req.compact_powder.model_dump(mode="json") if req.compact_powder else None,
    }


def describe(req: BeautyTryOnRequest) -> str:
    parts = []
    if req.compact_powder:
        p = req.compact_powder
        parts.append(f"powder {p.shade_name} {p.finish.value} {p.coverage}")
    if req.lipstick:
        lip = req.lipstick
        parts.append(f"lipstick {lip.shade_name} {lip.finish.value} {lip.intensity:.2f}")
    return " + ".join(parts) or "nothing"


def powder_tint(powder: CompactPowderConfig) -> Optional[str]:
    return None if powder.shade_id in TRANSLUCENT_IDS else powder.hex_tint


def apply_look(rgb: np.ndarray, pts: np.ndarray, lipstick: Optional[LipstickConfig],
               compact_powder: Optional[CompactPowderConfig]) -> np.ndarray:
    """Powder first (skin), then lipstick on top. Returns a new RGB uint8 image."""
    out = rgb
    if compact_powder:
        out = apply_powder(out, pts, powder_tint(compact_powder), POWDER_COVERAGE_T[compact_powder.coverage],
                           compact_powder.finish)
    if lipstick:
        out = ls.apply_lipstick(out, pts, lipstick.hex_color, lipstick.finish, lipstick.intensity)
    return out if out is not rgb else rgb.copy()


def locate_face(rgb: np.ndarray, mode: str) -> np.ndarray:
    """Face-mesh points; raises lipstick_service.LipsNotFound with a visitor-safe message."""
    # Powder alone needs a readable face, not readable lips, so the small-mouth limit is lipstick's only
    return ls.locate_face(rgb, require_lips=mode != MODE_POWDER)
