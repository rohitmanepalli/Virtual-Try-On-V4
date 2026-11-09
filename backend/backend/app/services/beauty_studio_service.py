"""
Studio portrait for a saved beauty look ("Save look" only — never for live previews).

Gemini re-lights the visitor's portrait as a retail vanity-studio photograph. Gemini redraws the whole face, so the
router then puts the visitor's real face back (face_preservation.restore_original_face) with its OWN skin tone and at
no more than MAX_FACE_UPSCALE enlargement, and applies the selected lipstick and powder locally on that real face. The
visitor therefore always sees their own face, skin tone and exactly the shade they picked in the preview; Gemini
contributes the light and backdrop. A render whose skin tone drifted (e.g. lightened) is not used: the save falls back
to the local look.
"""
import io
import os
import time
from typing import Optional

from PIL import Image

from app.config import get_gemini_api_key
from app.models import CompactPowderConfig, CompactPowderFinish, LipstickConfig, LipstickFinish
from app.services.gemini_cost import TRYON_MODEL, record_usage

# Off by default: Save look saves the local preview (the visitor's real photo, no Gemini call, no cost).
# BEAUTY_STUDIO_RENDER=true in .env turns the paid studio portrait back on.
BEAUTY_STUDIO_RENDER = os.getenv("BEAUTY_STUDIO_RENDER", "false").strip().lower() in ("1", "true", "yes", "on")
# 1K: the render's face then has about the camera close-up's own size, so the real face goes back without being
# enlarged (2K enlarged it 2-3x: smooth, textureless lips under a crisp lipstick edge looked painted on)
STUDIO_IMAGE_SIZE = os.getenv("BEAUTY_STUDIO_IMAGE_SIZE", "1K")
PORTRAIT_FACE_SPAN = 2.6  # portrait width / face width: head and shoulders
MAX_FACE_UPSCALE = 1.5  # enlarging the real face more than this loses the lip texture makeup is judged on

FINISH_TEXT = {
    LipstickFinish.MATTE: ("Matte", "Rich, velvety, even pigmentation with a soft diffused matte texture, crisp clean "
                                    "edges, and muted reflections."),
    LipstickFinish.SATIN: ("Creamy Satin", "Smooth moisturizing finish with a natural, subtle sheen across the fullness "
                                           "of the lower lip."),
    LipstickFinish.GLOSS: ("High-Shine Gloss", "Luminous, high-shine dimensional wet reflection with authentic specular "
                                               "highlights along the center of the lower lip and Cupid's bow."),
}
POWDER_FINISH_TEXT = {
    CompactPowderFinish.MATTE: ("Matte", "a fully shine-free, soft matte complexion"),
    CompactPowderFinish.NATURAL_VELVET: ("Natural Velvet", "a soft-focus velvet finish with a natural, healthy skin look"),
    CompactPowderFinish.LUMINOUS: ("Luminous", "a soft, lit-from-within glow on the high points of the face, never oily"),
}
COVERAGE_TEXT = {"sheer": "sheer, barely-there coverage", "medium": "medium, buildable coverage"}

PROMPT_HEAD = """You are an expert commercial beauty photographer and state-of-the-art virtual cosmetics engine specializing in luxury retail makeup try-on.

INPUT IMAGES:
- Image 1: A raw portrait photograph of a real human customer.

TASK:
Generate a single, ultra-realistic, master-grade studio portrait of the EXACT SAME PERSON from Image 1 with the specified cosmetic application, captured under professional retail vanity studio illumination.

[APPLIED COSMETICS SPECIFICATION]
{LIPSTICK_SPEC}
{POWDER_SPEC}

[PRIORITY #1 — 100% FACIAL IDENTITY & ANATOMICAL LOCK]
- Replicate the EXACT real person from Image 1 with 100% biometric fidelity.
- Strictly preserve their exact facial geometry: eye shape, eye color, iris patterns, eyebrows, nose structure, smile lines, cheeks, chin, jawline, ears, facial asymmetry, hairline, and exact hairstyle.
- ACCESSORIES: Preserve all eyeglasses (exact frame and lens), earrings, necklaces, piercings, bindis, and forehead marks exactly as they appear in Image 1.
- NEVER alter age, bone structure, or ethnicity.
- SKIN TONE LOCK: Keep the exact natural skin tone, depth and undertone of Image 1. NEVER lighten, brighten, whiten, fair or pink up the skin; the studio light must not change how light or dark the person's skin looks.
- NO AI PLASTIC DOLL SKIN: Do not blur or airbrush away pores. Retain authentic human skin texture.
- HEAD & FACE GEOMETRY LOCK: Keep the head at the SAME size, position, tilt and turn within the frame as in Image 1, with the SAME expression and the SAME mouth opening. Nothing may cover or cross the face.
"""

POWDER_RULES = """
[PRIORITY #{N} — COMPACT POWDER & COMPLEXION ENHANCEMENT RULES]
- POWDER PLACEMENT: Apply the powder exclusively to facial skin surfaces (forehead, T-zone, cheeks, nose, and chin).
- BEARD & FACIAL HAIR LOCK: Strictly DO NOT apply powder over beards, mustaches, goatees, stubble, sideburns, or hairlines. Facial hair must remain 100% natural, sharp, and clean with zero dusty white residue or powder build-up.
- EYE & EYEBROW EXCLUSION: Eyebrows, eyelashes, and eyelids must remain completely untouched, sharp, and distinct.
- SHINE CONTROL & FINISH:
  * Softly mattify distracting camera flash glare, oily reflections, and sweat sheen on the T-zone and forehead.
  * Render the skin with a soft-focus, finely-milled luxury powder finish that naturally evens out slight blotchiness while retaining real skin pores and micro-texture.
  * NEVER produce a chalky, ashy, dry, cakey, or mask-like appearance. The skin must look breathable and naturally perfected.
"""

LIPSTICK_RULES = """
[PRIORITY #{N} — LIPSTICK APPLICATION RULES]
- Apply color strictly within the natural vermilion borders of the lips.
- Preserve 3D lip curvature, vertical creases, and authentic texture.
- TEETH & INNER MOUTH: If the mouth is open, teeth and tongue must remain 100% clean and completely untouched by pigment.
"""

PROMPT_TAIL = """
[PRIORITY #{N} — PORTRAIT ELEVATION & VANITY STUDIO ILLUMINATION]
- Correct only an obvious green or blue camera cast; keep the skin exactly as light, dark and warm as in Image 1.
- Flattering, diffused frontal vanity lighting that showcases makeup tone accuracy.
- Clean, neutral off-white or light warm boutique studio backdrop (#F8F7F5).

[PRIORITY #{M} — MASTER PHOTOGRAPHIC RESOLUTION]
- Razor-sharp commercial beauty photography quality.
- Output ONLY the final single portrait photograph.
"""


def build_prompt(lipstick: Optional[LipstickConfig], powder: Optional[CompactPowderConfig],
                 powder_tinted: bool = True) -> str:
    """The studio prompt for any Beauty Studio mode; sections for a product that is not worn are left out."""
    if lipstick:
        finish_label, finish_desc = FINISH_TEXT[lipstick.finish]
        pct = round(lipstick.intensity * 100)
        opacity = f"Full {pct}% rich pigment" if pct >= 75 else f"Sheer {pct}% tint"
        lipstick_spec = (f"- Lipstick: Shade \"{lipstick.shade_name}\", target color {lipstick.hex_color.upper()}, "
                         f"{finish_label} finish ({finish_desc}), application opacity: {opacity}.")
    else:
        lipstick_spec = "- Lipstick: none. Leave the lips exactly as in Image 1, with no added color or gloss."
    if powder:
        p_label, p_desc = POWDER_FINISH_TEXT[powder.finish]
        color = f"shade color {powder.hex_tint.upper()}" if powder_tinted else "colorless, adds no tint"
        powder_spec = (f"- Compact powder: \"{powder.shade_name}\" ({color}), {p_label} finish ({p_desc}), "
                       f"{COVERAGE_TEXT[powder.coverage]}.")
    else:
        powder_spec = "- Compact powder: none. Leave the skin's tone and texture exactly as in Image 1."

    prompt = PROMPT_HEAD.format(LIPSTICK_SPEC=lipstick_spec, POWDER_SPEC=powder_spec)
    n = 2
    if powder:
        prompt += POWDER_RULES.replace("{N}", str(n))
        n += 1
    if lipstick:
        prompt += LIPSTICK_RULES.replace("{N}", str(n))
        n += 1
    return prompt + PROMPT_TAIL.replace("{N}", str(n)).replace("{M}", str(n + 1))


def portrait_crop(image: Image.Image, face_box: dict) -> Image.Image:
    """Head-and-shoulders 3:4 crop around the face (full-length captures would otherwise give Gemini a tiny face)."""
    w, h = image.size
    fx, fy, fw, fh = face_box["x"] * w, face_box["y"] * h, face_box["w"] * w, face_box["h"] * h
    cw = min(w, fw * PORTRAIT_FACE_SPAN)
    ch = min(h, cw * 4 / 3)
    cw = ch * 3 / 4
    cx, cy = fx + fw / 2, fy + fh * 0.62  # face slightly above centre, shoulders below
    left = int(max(0, min(w - cw, cx - cw / 2)))
    top = int(max(0, min(h - ch, cy - ch / 2)))
    return image.crop((left, top, left + int(cw), top + int(ch)))


class StudioRenderError(Exception):
    pass


def studio_render_available() -> bool:
    return BEAUTY_STUDIO_RENDER and bool(get_gemini_api_key())


def render_studio_portrait(portrait: Image.Image, prompt: str, session_id: Optional[str] = None) -> bytes:
    """One Gemini image call. Returns the raw render bytes; raises StudioRenderError on any failure."""
    from google import genai
    from google.genai import types

    key = get_gemini_api_key()
    if not key:
        raise StudioRenderError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=key)
    started = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=TRYON_MODEL,
            contents=[prompt, portrait.convert("RGB")],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(image_size=STUDIO_IMAGE_SIZE, aspect_ratio="3:4"),
            ),
        )
    except Exception as err:
        raise StudioRenderError(f"Gemini call failed: {err}") from err
    record_usage("beauty_studio_render", TRYON_MODEL, response, session_id=session_id,
                 duration_s=time.perf_counter() - started)

    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None) is not None:
        raise StudioRenderError(f"Gemini blocked the request: {feedback.block_reason}")
    parts = getattr(response, "parts", None)
    if not parts and getattr(response, "candidates", None):
        content = getattr(response.candidates[0], "content", None)
        parts = getattr(content, "parts", None)
    for part in parts or []:
        data = getattr(getattr(part, "inline_data", None), "data", None)
        if data:
            Image.open(io.BytesIO(data)).verify()  # a broken image is a failed render, not a crash later
            return data
    raise StudioRenderError("Gemini returned no image")
