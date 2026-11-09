"""Beauty Studio endpoints: lipstick only, compact powder only, or both. Shares no state with the clothing try-on.

Live previews are local and free. "Save look" (include_qr) additionally makes one studio portrait with Gemini (see
beauty_studio_service), then puts the visitor's real face back and applies the makeup locally on it; whenever that
cannot be done cleanly, the local result is saved instead.
"""
import asyncio
import base64
import io
import os
import time
import uuid
from typing import List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image

from app.config import get_public_base_url
from app.services import download_store
from app.models import BeautyTryOnRequest, BeautyTryOnResponse, LipstickFinish, LipstickShade, PowderShade
from app.services import beauty_service
from app.services import beauty_studio_service as studio
from app.services import lipstick_service as ls
from app.services.face_preservation import restore_original_face
from app.services.image_utils import apply_watermark, clean_and_decode_base64
from app.services.lipstick_service import LipsNotFound
from app.services.qr_service import generate_qr_data_uri

router = APIRouter()

# Tuned for Indian skin tones (light-wheatish to deep; mostly warm, golden and olive undertones)
_M, _S, _G = LipstickFinish.MATTE, LipstickFinish.SATIN, LipstickFinish.GLOSS
SHADES: List[LipstickShade] = [
    LipstickShade(id="dusty_rose", name="Dusty Rose", category="Roses", hex_color="#A85A63", finish_default=_M),
    LipstickShade(id="deep_rose", name="Deep Rose", category="Roses", hex_color="#9E3F55", finish_default=_S),
    LipstickShade(id="terracotta", name="Terracotta", category="Reds", hex_color="#A54E37", finish_default=_M),
    LipstickShade(id="brick_red", name="Brick Red", category="Reds", hex_color="#8F2E25", finish_default=_M),
    LipstickShade(id="classic_red", name="Classic Red", category="Reds", hex_color="#A3202E", finish_default=_S),
    LipstickShade(id="sindoor_red", name="Sindoor Red", category="Reds", hex_color="#B7322A", finish_default=_G),
    LipstickShade(id="berry", name="Berry", category="Berries & Wines", hex_color="#7E2645", finish_default=_S),
    LipstickShade(id="maroon", name="Maroon", category="Berries & Wines", hex_color="#6B1E24", finish_default=_M),
    LipstickShade(id="wine", name="Wine", category="Berries & Wines", hex_color="#5A1A28", finish_default=_M),
    LipstickShade(id="plum", name="Plum", category="Berries & Wines", hex_color="#5E2236", finish_default=_S),
    LipstickShade(id="chocolate_brown", name="Chocolate Brown", category="Browns", hex_color="#5C3328",
                  finish_default=_M),
]

POWDERS: List[PowderShade] = [
    PowderShade(id="warm_ivory", name="Warm Ivory", hex_tint="#E8C4A0"),
    PowderShade(id="light_beige", name="Light Beige", hex_tint="#D9AE85"),
    PowderShade(id="natural_honey", name="Natural Honey", hex_tint="#C99A6E"),
    PowderShade(id="golden_medium", name="Golden Medium", hex_tint="#B98858"),
    PowderShade(id="caramel", name="Caramel", hex_tint="#A6744A"),
    PowderShade(id="warm_tan", name="Warm Tan", hex_tint="#93633F"),
    PowderShade(id="deep_bronze", name="Deep Bronze", hex_tint="#7A4F33"),
    PowderShade(id="espresso", name="Espresso", hex_tint="#5E3B27"),
]

_UNREADABLE = "The photo could not be read. Please retake it."


@router.get("/api/beauty/shades", response_model=List[LipstickShade])
async def list_shades():
    return SHADES


@router.get("/api/beauty/powders", response_model=List[PowderShade])
async def list_powders():
    return POWDERS


def _encode(rgb, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return buf.getvalue()


def _save_download(rgb, session_id=None) -> str:
    """Watermarked download copy in memory (from a lossless encode, so it is JPEG-compressed once) and its QR code."""
    image_id = str(uuid.uuid4())
    download_store.put(image_id, apply_watermark(_encode(rgb, "PNG")), session_id)
    return generate_qr_data_uri(f"{get_public_base_url()}/download/{image_id}")


def _failure(data: BeautyTryOnRequest, message: str) -> BeautyTryOnResponse:
    return BeautyTryOnResponse(success=False, mode=beauty_service.look_mode(data), result_image="",
                               active_look=beauty_service.active_look(data), message=message)


def _local_look(data: BeautyTryOnRequest) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(result, face points, input image) of the instant local look. Raises LipsNotFound, or ValueError when the
    photo cannot be read."""
    try:
        image = clean_and_decode_base64(data.image)
        image.load()
        rgb = np.ascontiguousarray(np.array(image.convert("RGB")))
    except Exception as err:
        raise ValueError("unreadable image") from err
    pts = beauty_service.locate_face(rgb, beauty_service.look_mode(data))
    return beauty_service.apply_look(rgb, pts, data.lipstick, data.compact_powder), pts, rgb


def _response(data: BeautyTryOnRequest, result: np.ndarray, pts: np.ndarray, qr: Optional[str],
              studio_render: bool, started: float) -> BeautyTryOnResponse:
    jpeg = _encode(result, "JPEG")
    h, w = result.shape[:2]
    mode = beauty_service.look_mode(data)
    print(f"[BEAUTY] {mode} | {beauty_service.describe(data)} | {'studio' if studio_render else 'local'} | "
          f"qr {'yes' if qr else 'no'} | session {(data.session_id or '-')[:8]} | "
          f"{(time.perf_counter() - started) * 1000:.0f} ms", flush=True)
    return BeautyTryOnResponse(
        success=True,
        mode=mode,
        result_image=f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode('utf-8')}",
        qr_code_base64=qr,
        active_look=beauty_service.active_look(data),
        face_box=ls.face_box(pts, w, h),
        studio_render=studio_render,
    )


def _run_tryon(data: BeautyTryOnRequest) -> BeautyTryOnResponse:
    started = time.perf_counter()
    try:
        result, pts, _ = _local_look(data)
    except LipsNotFound as err:
        return _failure(data, str(err))
    except ValueError:
        return _failure(data, _UNREADABLE)
    return _response(data, result, pts, None, False, started)


async def _studio_look(data: BeautyTryOnRequest, local_pts: np.ndarray,
                       source_rgb: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(result, face points) of the studio portrait with the visitor's real face, or None to save the local look."""
    sid = (data.session_id or "-")[:8]
    h, w = source_rgb.shape[:2]
    portrait = studio.portrait_crop(Image.fromarray(source_rgb), ls.face_box(local_pts, w, h))
    tinted = data.compact_powder is None or beauty_service.powder_tint(data.compact_powder) is not None
    prompt = studio.build_prompt(data.lipstick, data.compact_powder, tinted)
    try:
        render = await asyncio.to_thread(studio.render_studio_portrait, portrait, prompt, data.session_id)
    except studio.StudioRenderError as err:
        print(f"[BEAUTY STUDIO] render failed ({err}); saving the local look | session {sid}", flush=True)
        return None
    # On the event loop thread, like the clothing pipeline: restore uses the shared face detector, so the two never run
    # it at the same time. Gemini redrew the face; the visitor's real one goes back with its own skin tone and barely
    # enlarged. A render whose skin is off that tone (e.g. lightened), or a face too small to enlarge, is refused.
    restored_bytes, check = restore_original_face(portrait, render, keep_capture_tone=True,
                                                  max_upscale=studio.MAX_FACE_UPSCALE)
    if not check.restored:
        print(f"[BEAUTY STUDIO] real face not restorable ({check.reason}); saving the local look | session {sid}",
              flush=True)
        return None
    try:
        restored = np.ascontiguousarray(np.array(Image.open(io.BytesIO(restored_bytes)).convert("RGB")))
        mode = beauty_service.look_mode(data)
        pts = await asyncio.to_thread(beauty_service.locate_face, restored, mode)
        result = await asyncio.to_thread(beauty_service.apply_look, restored, pts, data.lipstick, data.compact_powder)
    except (LipsNotFound, ValueError) as err:
        print(f"[BEAUTY STUDIO] makeup on the studio portrait failed ({err}); saving the local look | session {sid}",
              flush=True)
        return None
    print(f"[BEAUTY STUDIO] studio portrait ready | face match {check.score:.3f} | session {sid}", flush=True)
    return result, pts


async def _save_look(data: BeautyTryOnRequest) -> BeautyTryOnResponse:
    started = time.perf_counter()
    try:
        local, pts, source = await asyncio.to_thread(_local_look, data)
    except LipsNotFound as err:
        return _failure(data, str(err))
    except ValueError:
        return _failure(data, _UNREADABLE)
    studio_result = await _studio_look(data, pts, source) if studio.studio_render_available() else None
    result, pts = studio_result if studio_result else (local, pts)
    qr = await asyncio.to_thread(_save_download, result, data.session_id)
    return _response(data, result, pts, qr, studio_result is not None, started)


@router.post("/api/beauty/tryon", response_model=BeautyTryOnResponse)
async def beauty_tryon(data: BeautyTryOnRequest):
    if data.lipstick is None and data.compact_powder is None:
        raise HTTPException(status_code=422, detail="Choose a lipstick, a compact powder, or both.")
    if data.include_qr:
        return await _save_look(data)
    # CPU-bound (decode, face mesh, encode): keep it off the event loop so clothing requests are not stalled
    return await asyncio.to_thread(_run_tryon, data)
