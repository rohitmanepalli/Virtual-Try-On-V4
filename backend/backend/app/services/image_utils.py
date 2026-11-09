import base64
import io
import os
import re
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from app.config import MAX_IMAGE_SIZE


def clean_and_decode_base64(base64_string: str) -> Image.Image:
    """Purges browser data headers and standardizes image string data formats."""
    cleaned_data = re.sub(r'^data:image/.+;base64,', '', base64_string).strip()
    missing_padding = len(cleaned_data) % 4
    if missing_padding:
        cleaned_data += '=' * (4 - missing_padding)
    return Image.open(io.BytesIO(base64.b64decode(cleaned_data)))


def downscale(image: Image.Image) -> None:
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)


FINAL_JPEG_QUALITY = 95

# Fixed frame for every render, the same numbers the prompt's [FRAMING] section asks the model for
FRAME_ASPECT = 0.75  # 3:4
FRAME_TOP = 0.05  # top of the head below the top edge, as a share of the height
FRAME_BOTTOM = 0.04  # soles above the bottom edge
FRAME_SIDE_MAX = 0.96  # widest the person may be, as a share of the width


def normalize_framing(image_bytes: bytes, person_mask: Optional[np.ndarray], decoded: Optional[Image.Image] = None) -> bytes:
    """
    Re-frames a render so the person always has the same scale and position, whatever zoom the model chose: head
    FRAME_TOP below the top edge, feet FRAME_BOTTOM above the bottom, centred, on a 3:4 canvas as tall as the render.
    Only scales and pads (with the render's own backdrop colour), never crops the person. person_mask: the render's
    person segmentation at any resolution; returns the input unchanged without one. decoded: image_bytes already
    decoded, when the caller has it.
    """
    if person_mask is None:
        return image_bytes
    mask = np.squeeze(person_mask) > 0.5
    if mask.ndim != 2 or mask.sum() < 0.01 * mask.size:
        return image_bytes
    img = (decoded if decoded is not None else Image.open(io.BytesIO(image_bytes))).convert("RGB")
    w, h = img.size
    ys, xs = np.nonzero(mask)
    mh, mw = mask.shape
    x1, x2 = xs.min() / mw * w, (xs.max() + 1) / mw * w
    y1, y2 = ys.min() / mh * h, (ys.max() + 1) / mh * h

    out_h, out_w = h, round(h * FRAME_ASPECT)
    band = (1.0 - FRAME_TOP - FRAME_BOTTOM) * out_h
    scale = min(band / (y2 - y1), FRAME_SIDE_MAX * out_w / (x2 - x1))
    resized = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)

    arr = np.asarray(img)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    canvas = Image.new("RGB", (out_w, out_h), tuple(int(c) for c in np.median(border, axis=0)))
    top = FRAME_TOP * out_h + (band - (y2 - y1) * scale) / 2.0  # centred in the band when the width limited the scale
    left = out_w / 2.0 - (x1 + x2) / 2.0 * scale
    canvas.paste(resized, (round(left), round(top - y1 * scale)))
    buf = io.BytesIO()
    # Lossless and uncompressed (PNG takes seconds at this size): finalize_result_image encodes the JPEG once
    canvas.save(buf, format="BMP")
    return buf.getvalue()


def finalize_result_image(image_bytes: bytes) -> Tuple[bytes, Tuple[int, int]]:
    """
    The single final encode of a finished try-on image: validated RGB JPEG at full resolution.

    - Never resizes.
    - Drops an alpha channel (flattened onto white; logged when it was not fully opaque).
    - Returns the input bytes untouched when they already are an RGB JPEG, so nothing is
      compressed twice. Lossless intermediates (PNG) are encoded exactly once here.
    Returns (jpeg_bytes, (width, height)).
    """
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    size = img.size
    if img.format == "JPEG" and img.mode == "RGB":
        return image_bytes, size

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 255:
            print(f"[FINAL IMAGE] Unexpected transparency in {img.mode} result; flattened onto white.")
        flat = Image.new("RGB", size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=FINAL_JPEG_QUALITY, subsampling=0)
    return buf.getvalue(), size


def apply_watermark(generated_image_bytes: bytes) -> bytes:
    """Returns the download copy: the image with the logo watermark, as JPEG. Works in memory; writes no file.

    Falls back to a JPEG without the watermark if watermarking fails.
    """
    try:
        generated_img = Image.open(io.BytesIO(generated_image_bytes)).convert("RGBA")

        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).convert("RGBA")

            # Calculate new logo size (e.g., 15% of the generated image width)
            target_logo_width = int(generated_img.width * 0.15)
            aspect_ratio = logo_img.height / logo_img.width
            target_logo_height = int(target_logo_width * aspect_ratio)

            logo_resized = logo_img.resize((target_logo_width, target_logo_height), Image.Resampling.LANCZOS)

            # Calculate position (Top Right with some padding)
            padding = int(generated_img.width * 0.03)  # 3% padding
            position_x = generated_img.width - target_logo_width - padding
            position_y = padding

            # Paste logo using itself as a mask for transparency
            generated_img.paste(logo_resized, (position_x, position_y), mask=logo_resized)

        # Convert back to RGB to save as JPEG (4:4:4 chroma, high quality: this is the downloadable 4K copy)
        final_img = generated_img.convert("RGB")
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=FINAL_JPEG_QUALITY, subsampling=0)
        return buf.getvalue()

    except Exception as e:
        print(f"[WARNING] Failed to apply watermark: {e}")
        return finalize_result_image(generated_image_bytes)[0]
