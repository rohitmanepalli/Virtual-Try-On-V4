import hashlib
import io
import json
import math
import os
import threading
import time
from typing import Dict, Optional, Tuple
from fastapi import HTTPException
from PIL import Image

from google import genai
from google.genai import types

from app.config import TRYON_IMAGE_SIZE, get_gemini_api_key
from app.models import BodyProfile, ClothAnalysis
from app.services import capture_cache, request_timing
from app.services.face_preservation import restore_original_face
from app.services.fit_drape import body_drape_text, size_fit_section
from app.services.garment_coverage import (
    OWN_BOTTOMS, SLEEVE_LENGTHS, TOP_LENGTHS, CaptureCoverage, coverage_section, normalize_length,
)
from app.services.gemini_cost import ANALYSIS_MODEL, TRYON_MODEL, record_usage

PROMPT_TEXT = (
    "# VIRTUAL TRY-ON\n"
    "\n"
    "Image 1: real customer photograph. Image 2: retail catalog garment.\n"
    "Generate one photorealistic full-length photograph of the exact person in Image 1 wearing the exact garment in Image 2.\n"
    "\n"
    "### 1. Face and head (locked)\n"
    "- Keep the exact identity: facial features, expression, skin tone and texture, hairstyle, facial hair and glasses. Never beautify, smooth or regenerate them.\n"
    "- Keep the head's size relative to the body, its position over the shoulders, its tilt and rotation, and the neck alignment exactly as in Image 1. The real face is composited back onto this image afterwards, so they must line up.\n"
    "\n"
    "### 2. Body (locked)\n"
    "- Keep the exact proportions, height, shoulder width, waist, stomach and natural posture.\n"
    "- Never slim, enlarge, flatten, reshape or idealize the body. The garment adapts to the body, never the other way around.\n"
    "\n"
    "### 3. Garment\n"
    "- Reproduce the exact color, design, print, embroidery, fabric, neckline, collar, sleeves, buttons and accessories of Image 2. Never invent, simplify or remove any of them.\n"
    "- Fully replace the person's clothing and sleeves: their own clothing never shows past the selected garment's own sleeves or hem.\n"
    "- Inner clothing stays visible only where it naturally shows through the opening of an open garment.\n"
    "- When Image 2 is a complete outfit, its matching bottom replaces the person's own lower garment.\n"
    "\n"
    "### 4. Size and fit\n"
    "- Follow the size and fit sections below, which come from this person's body and the selected size. A size letter alone means nothing: a Medium may be tight on a larger person, and Large or XXL must show the extra room they really give this person.\n"
    "- Only the garment changes between sizes, never the body.\n"
    "- Natural shoulder placement, realistic folds, tension and drape. No floating, distorted or painted-on clothing.\n"
    "\n"
    "### 5. Personal features\n"
    "- Keep every bindi (regardless of gender), tattoo, piece of jewelry, watch, scar and other visible mark exactly where it is. Never invent or remove one.\n"
    "\n"
    "### 6. Hands, feet and anatomy\n"
    "- Keep both arms, hands, legs and feet, with the arms and hands in their original positions and five natural fingers on each hand. No extra, missing or distorted limbs or fingers.\n"
    "- Sleeves, cuffs, dupattas, pallus and hems must never completely hide a hand or foot.\n"
    "- Show both hands completely, wrist to fingertips with the thumb and all fingers: never cropped, merged with clothing, behind the body or in a pocket. If a hand is hidden in Image 1, bring it naturally into view.\n"
    "- A dupatta, pallu, stole or shawl keeps its design but hangs clear of both hands and forearms, even where Image 2 shows it draped over a sleeve.\n"
    "- If Image 1 crops the lower body, continue it naturally down to the feet.\n"
    "\n"
    "### 7. Framing (identical in every image)\n"
    "- Fixed 3:4 portrait, person centered and whole from head to feet: top of the head 5% below the top edge, soles 4% above the bottom edge.\n"
    "- Keep the same camera distance and subject scale every time, whatever the garment or size. Never zoom or crop.\n"
    "\n"
    "### 8. Image quality\n"
    "- Seamless clean white studio background filling the frame, with nothing from the room in Image 1.\n"
    "- Professional soft studio lighting, a subtle floor shadow and accurate garment colors.\n"
    "- Accurate fabric texture, natural folds, sharp garment details and natural skin texture.\n"
    "- No AI artifacts, unwanted design changes, duplicated accessories, text, watermark or borders. Output one image only.\n"
    "\n"
    "**HIGHEST PRIORITY:** the exact person and body of Image 1, wearing the exact garment of Image 2.\n"
)

import cv2
import numpy as np

FACE_NOT_RESTORED_MESSAGE = "We couldn't keep your face clear in this preview. Please face the camera and try again."

_client_lock = threading.Lock()
_shared_client: Tuple[Optional[tuple], Optional[genai.Client]] = (None, None)


def _get_client(api_key: str) -> genai.Client:
    """One shared client: building one costs about 0.7s (it loads the system certificates)."""
    global _shared_client
    identity = (genai.Client, api_key)
    with _client_lock:
        if _shared_client[0] != identity:
            _shared_client = (identity, genai.Client(api_key=api_key))
        return _shared_client[1]


# In-memory cache for garment cloth analysis keyed by thumbnail hash
_CLOTH_ANALYSIS_CACHE: Dict[str, ClothAnalysis] = {}

# Pre-registered catalog profiles for instant zero-latency recognition
_PRE_REGISTERED_CATALOG: Dict[str, ClothAnalysis] = {
    # blazer1: Grey textured tailored blazer
    "e9d7a8f79fee6498": ClothAnalysis(
        category="blazer",
        garment_name="Men's Textured Grey Tailored Blazer",
        fabric_type="Structured medium-to-heavyweight textured tweed/herringbone suiting fabric with subtle vertical ribbed grain, interior lining, and canvas chest interlining",
        structure_level="rigid_tailored",
        is_outerwear=True,
        sleeve_length="full",
        top_length="hip",
        sleeve_behavior="Long sleeves must hang in a clean, smooth, structured cylindrical drape with zero horizontal accordion folds, wrinkles, ripples, or bunching. Keep normal, tailored, straight arm hang down to the cuffs.",
        tryon_guidelines="This garment is an outerwear blazer. It must sit cleanly as a structured tailored jacket over any inner shirt/chest, with distinct fabric thickness, structured shoulder pads, and crisp notch lapels. Strictly DO NOT render as shirt-on-shirt. Preserve the authentic fabric thickness, tweed texture, welt chest pocket, and flap waist pockets."
    ),
    # blazer2: Dark/navy tailored formal blazer
    "5eccaa5d7c434a18": ClothAnalysis(
        category="blazer",
        garment_name="Men's Classic Tailored Formal Blazer",
        fabric_type="Structured fine wool suiting blend with smooth finish, full satin interior lining, and canvas chest piece",
        structure_level="rigid_tailored",
        is_outerwear=True,
        sleeve_length="full",
        top_length="hip",
        sleeve_behavior="Smooth, clean, structured tailored cylindrical sleeve hang down both arms. Zero ripples, zero crumples, zero bunched folds on the arms.",
        tryon_guidelines="Structured outerwear blazer with defined shoulder pads, crisp notch lapels, and tailored waist contour. Layer cleanly over inner shirt without blending or looking like shirt-on-shirt."
    ),
    # male1: Casual button-down shirt
    "da8dff6edab84939": ClothAnalysis(
        category="shirt",
        garment_name="Men's Classic Casual Button-Down Shirt",
        fabric_type="Crisp woven cotton shirt fabric with clean collar and button placket",
        structure_level="semi_structured",
        is_outerwear=False,
        sleeve_length="full",
        top_length="hip",
        sleeve_behavior="Natural buttoned shirt sleeve drape ending cleanly at the wrists.",
        tryon_guidelines="Fit cleanly across torso and shoulders with natural drape."
    ),
    # kurti1: White/cream straight-cut Kurti with floral block-print center panel and palazzo
    "5184981f962a5400": ClothAnalysis(
        category="kurti",
        garment_name="Women's Floral Block-Print Panel Straight Kurti Set",
        fabric_type="Pure cotton/chanderi soft drape fabric in off-white/cream with vibrant floral block-print center panel and coordinating cream palazzo pants",
        structure_level="soft_drape",
        is_outerwear=False,
        sleeve_length="three_quarter",
        top_length="below_knee",
        sleeve_behavior="Clean, relaxed three-quarter (quarter-length) sleeves ending at mid-forearm with matching printed cuffs, hanging with comfortable natural ease.",
        tryon_guidelines="This garment is a traditional Indian straight-cut Kurti paired with coordinating wide-leg palazzo pants. The Kurti MUST hang straight from shoulder to hem with consistent width down the torso and hips—do NOT taper or pull in at the waist like a Western fitted dress. The hemline extends from knee to calf length with traditional crisp side slits (chaak). Accurately preserve the floral block-print center panel, mandarin notched neckline, and coordinating cream palazzo pants."
    ),
    # kurti1 (alternate hash alias)
    "9afdf5217b9abb40": ClothAnalysis(
        category="kurti",
        garment_name="Women's Floral Block-Print Panel Straight Kurti Set",
        fabric_type="Pure cotton/chanderi soft drape fabric in off-white/cream with vibrant floral block-print center panel and coordinating cream palazzo pants",
        structure_level="soft_drape",
        is_outerwear=False,
        sleeve_length="three_quarter",
        top_length="below_knee",
        sleeve_behavior="Clean, relaxed three-quarter (quarter-length) sleeves ending at mid-forearm with matching printed cuffs, hanging with comfortable natural ease.",
        tryon_guidelines="This garment is a traditional Indian straight-cut Kurti paired with coordinating wide-leg palazzo pants. The Kurti MUST hang straight from shoulder to hem with consistent width down the torso and hips—do NOT taper or pull in at the waist like a Western fitted dress. The hemline extends from knee to calf length with traditional crisp side slits (chaak). Accurately preserve the floral block-print center panel, mandarin notched neckline, and coordinating cream palazzo pants."
    ),
    # kurta1: Men's peach gold silk designer kurta pajama set
    "a6b359902c9455cc": ClothAnalysis(
        category="kurta",
        garment_name="Men's Designer Raw Silk Embroidered Kurta Pajama Set",
        fabric_type="Lustrous peach-gold raw silk with intricate thread embroidery on mandarin collar, front button placket, and sleeve cuffs, paired with crisp off-white tailored trousers",
        structure_level="semi_structured",
        is_outerwear=False,
        sleeve_length="full",
        top_length="knee",
        sleeve_behavior="Clean, tailored full-length sleeves with embroidered cuffs draping smoothly along arms to wrists without awkward bunching.",
        tryon_guidelines="This garment is an elegant traditional Indian men's designer raw silk kurta pajama set. The kurta extends to knee length with traditional crisp side slits (chaak) at hip/thigh height. Accurately preserve the peach-gold silk texture, intricate placket embroidery, mandarin band collar, and pair with crisp off-white bottom trousers."
    ),
    # sherwani1: Sage-green embroidered sherwani with ivory draped dhoti
    "9f770cf15fd9649e": ClothAnalysis(
        category="sherwani",
        garment_name="Men's Sage-Green Embroidered Sherwani with Ivory Draped Dhoti",
        fabric_type="Structured sage-green silk-blend suiting with tone-on-tone floral thread embroidery all over and a dense embroidered border at the hem, cuffs and collar, interlined through the chest and shoulders; soft ivory silk-blend dhoti",
        structure_level="rigid_tailored",
        is_outerwear=False,
        sleeve_length="full",
        top_length="knee",
        sleeve_behavior="Structured full-length sleeves hanging as clean straight tubes to the wrist with embroidered cuffs; no bunching or accordion folds.",
        tryon_guidelines="This is a traditional Indian men's sherwani (bandhgala-style long jacket) worn with an ivory dhoti. The jacket is structured through the shoulders and chest like tailoring, has a mandarin band collar closed at the neck, a single row of covered buttons down the centre front, and falls straight to the knee, where the fronts part slightly below the last button. No inner clothing shows at the collar or front. Preserve the sage-green shade, the floral embroidery and its denser borders, and the beige pocket square in the left chest pocket. The dhoti replaces the visitor's own trousers: soft ivory fabric draped from the waist in deep loose cowl folds and gathered at the ankles."
    ),
    # kurta_dhoti1: Plain white cotton kurta with white front-pleated dhoti, maroon border
    "65a832f07fadc2af": ClothAnalysis(
        category="kurta_dhoti",
        garment_name="Men's White Cotton Kurta with Traditional Front-Pleated Dhoti",
        fabric_type="Plain white fine cotton kurta with a matte weave; white cotton dhoti (veshti) with a narrow woven maroon border",
        structure_level="semi_structured",
        is_outerwear=False,
        sleeve_length="three_quarter",
        top_length="thigh",
        sleeve_behavior="Three-quarter sleeves ending at mid-forearm with plain hems; the forearms and hands show below them, and the visitor's own long sleeves must not appear under or below them.",
        tryon_guidelines="This is a traditional Indian men's white cotton kurta worn with a white dhoti. The kurta is collarless with a round neckline and a short button placket, falls straight to mid-thigh with small side slits, and hangs over the top of the dhoti. The dhoti replaces the visitor's own trousers: tied at the waist in the classic front-pleated style, with neat vertical pleats at the centre front falling to the ankles and the maroon border visible along the pleat edge and hem. Keep both pieces plain bright white with natural cotton texture; no embroidery or print."
    ),
    # salwar_suit1: Sage/teal-grey zari-embroidered salwar kameez with sheer teal dupatta
    "d1893e3bbb81c513": ClothAnalysis(
        category="salwar_suit",
        garment_name="Women's Zari-Embroidered Salwar Suit with Sheer Dupatta",
        fabric_type="Soft silky sage / teal-grey kameez and salwar with heavy gold and white zari embroidery; sheer teal chiffon dupatta with scattered gold sequins and a tasseled gold gota-patti trim",
        structure_level="soft_drape",
        is_outerwear=False,
        sleeve_length="full",
        top_length="knee",
        sleeve_behavior="Long straight sleeves ending at the wrist with wide gold zari-embroidered cuffs, hanging with gentle ease.",
        tryon_guidelines="This is a traditional Indian women's three-piece salwar suit. The kameez is straight cut to just above the knee with a V-neckline framed by a wide gold zari embroidered yoke, embroidered cuffs, and a deep embroidered hem border with small gold tassels; it hangs straight from the shoulders without tapering at the waist. The salwar replaces the visitor's own trousers: the same sage / teal-grey fabric, loose and gathered at the waist, falling in soft folds to an embroidered ankle border. The sheer teal dupatta, with gold sequins and tasseled gold trim on all edges, is draped over the left shoulder and hangs down the front and back on that side, keeping the neckline embroidery visible and never covering the face; it replaces any scarf or dupatta the visitor wears in Image 1. STRICT HAND VISIBILITY FOR THIS OUTFIT: full-length, straight-on front view with both arms separated slightly from the torso and fully visible from shoulder to fingertips, each palm, wrist, thumb and all five relaxed fingers clear, with natural arm proportions. Image 2 shows the dupatta lying over the wearer's left sleeve and cuff: do NOT copy that part of the drape. From the wearer's left shoulder (viewer's right) the dupatta falls in front of the chest and torso on that side, inside the arm, so the whole left sleeve, cuff and hand stay outside it and uncovered; its fabric, colour, sequins and gold tasseled trim stay exactly as in Image 2. The wearer's right arm and hand (viewer's left) are equally clear. Change nothing else about the kameez, salwar, embroidery, neckline or borders."
    ),
    # kurta_pajama1: Magenta-pink long kurta with black button loops, white straight-cut pants
    "b4597b0ae092048c": ClothAnalysis(
        category="kurta_pajama",
        garment_name="Men's Magenta Kurta with White Straight-Cut Pants",
        fabric_type="Solid magenta-pink cotton with a fine matte weave; plain white cotton straight-cut pants",
        structure_level="semi_structured",
        is_outerwear=False,
        sleeve_length="full",
        top_length="knee",
        sleeve_behavior="Full-length sleeves ending at the wrist in neat buttoned cuffs, hanging smoothly along the arms.",
        tryon_guidelines="This is a traditional Indian men's long kurta worn with white pants. The kurta is solid magenta-pink, straight cut and long, with side slits at the hem, a mandarin band collar, and a short front placket of small round black buttons held by decorative black thread loops. The white pants replace the visitor's own trousers: plain, straight-cut and slim, visible below the kurta hem down to the ankles. Keep the kurta a plain solid colour with no embroidery or print."
    ),
    # lehenga1: Multicolour (orange / blue / magenta) embroidered bridal lehenga with dupatta
    "a8f933cee637ac41": ClothAnalysis(
        category="lehenga",
        garment_name="Women's Multicolour Embroidered Bridal Lehenga",
        fabric_type="Raw silk blouse and flared skirt in panels of bright orange, royal blue and magenta with multicolour floral thread embroidery, mirror-work and a wide gold embroidered hem border; sheer orange and magenta dupatta with a jeweled gold border",
        structure_level="semi_structured",
        is_outerwear=False,
        sleeve_length="short",
        top_length="cropped",
        sleeve_behavior="Short fitted sleeves ending above the elbow with embroidered gold edges.",
        tryon_guidelines="This is a traditional Indian women's three-piece bridal lehenga. The fitted blouse (choli) has a V-neckline and short sleeves and ends at the midriff; it replaces the visitor's own top. The floor-length skirt replaces the visitor's own lower garment: gathered at the waist and flaring wide to the floor in alternating orange, royal blue and magenta panels with floral embroidery and mirror-work, and a wide gold hem border. The sheer dupatta with its jeweled gold border is draped diagonally across the body from the left shoulder to the right hip, keeping the blouse neckline visible and never covering the face; it replaces any scarf or dupatta the visitor wears in Image 1."
    ),
    # royal_sherwani1: Layered gold sherwani, inner kurta, churidar, stole and maroon turban
    "3a88cdc694342135": ClothAnalysis(
        category="royal_sherwani",
        garment_name="Men's Layered Gold Embroidered Sherwani with Stole and Turban",
        fabric_type="Structured gold / beige silk sherwani with heavy zardozi embroidered borders over a pale gold inner kurta with gold buti embroidery; beige churidar; cream stole with embroidered paisley borders; maroon polka-dot turban with a gold jeweled brooch",
        structure_level="rigid_tailored",
        is_outerwear=False,
        sleeve_length="full",
        top_length="below_knee",
        sleeve_behavior="Long structured sleeves with embroidered cuffs, hanging as clean straight tubes to the wrist.",
        tryon_guidelines="This is a traditional Indian men's layered royal wedding outfit. The long gold sherwani is worn open at the front over the matching inner kurta, which shows down the centre front between the embroidered front edges; the sherwani and inner kurta together replace the visitor's own top, so none of the visitor's own clothing shows. The beige churidar replaces the visitor's own trousers: close-fitting down the calves and gathered at the ankles. The cream stole is draped around the neck and hangs down the front over both shoulders. The maroon polka-dot turban with its gold brooch sits on top of the head above the hairline, covering the hair; it never covers the forehead, eyebrows, eyes or face, and the visitor's glasses stay exactly as they are."
    ),
    # kerala_saree1: Cream kasavu saree with gold zari border, dark green blouse, gold temple necklace
    "0b3cddb26cce9ddb": ClothAnalysis(
        category="saree",
        garment_name="Women's Kerala Kasavu Saree with Green Blouse",
        fabric_type="Plain off-white / cream handloom cotton saree with a wide gold zari border and gold-banded pallu; dark bottle-green silk blouse",
        structure_level="semi_structured",
        is_outerwear=False,
        sleeve_length="short",
        top_length="cropped",
        sleeve_behavior="Short fitted blouse sleeves ending above the elbow.",
        tryon_guidelines="This is a traditional Kerala kasavu saree outfit. The fitted dark green blouse with short sleeves replaces the visitor's own top. The cream saree replaces the visitor's own lower garment: wrapped at the waist with neat vertical pleats at the centre front falling straight to the floor, the gold zari border along its edges and hem, and the gold-banded pallu taken diagonally across the chest and over the left shoulder. The gold temple necklace shown in Image 2 is part of this outfit and rests at the neckline; the pallu never covers the face."
    ),
    # mundu_shirt1: Cream full-sleeve shirt tucked into a gold-bordered kasavu mundu, with matching shawl
    "408b08e38754cb4f": ClothAnalysis(
        category="mundu_shirt",
        garment_name="Men's Cream Shirt with Gold-Bordered Kasavu Mundu and Shawl",
        fabric_type="Plain cream / off-white cotton shirt; cream cotton mundu and shawl with wide gold kasavu borders",
        structure_level="semi_structured",
        is_outerwear=False,
        sleeve_length="full",
        top_length="waist",
        sleeve_behavior="Full-length shirt sleeves ending at the wrist in buttoned cuffs.",
        tryon_guidelines="This is a traditional Kerala men's outfit. The plain cream collared full-sleeve shirt replaces the visitor's own top and is tucked in at the waist. The cream mundu replaces the visitor's own trousers: wrapped at the waist and falling straight to the ankles, with the wide gold kasavu border running down the front overlap edge and along the hem. The matching gold-bordered shawl is draped over the left shoulder and hangs down the front on that side to the thigh."
    ),
}


def get_dress_hash(dress_img: Image.Image) -> str:
    """Computes a standardized thumbnail hash for perceptual image matching."""
    thumb = dress_img.convert("RGB").resize((128, 128))
    return hashlib.sha256(thumb.tobytes()).hexdigest()


def analyze_cloth(
    dress_img: Image.Image,
    client: Optional[genai.Client] = None
) -> ClothAnalysis:
    """
    Performs deep garment analysis to detect category, fabric weight, structure,
    outerwear layering, and sleeve drape behavior.
    Results are cached in-memory for instant 0ms retrieval on subsequent calls.
    """
    thumb_hash = get_dress_hash(dress_img)
    prefix_16 = thumb_hash[:16]

    # 1. Check in-memory cache
    if thumb_hash in _CLOTH_ANALYSIS_CACHE:
        print(f"[CLOTH ANALYSIS] Cache hit for garment hash {prefix_16} -> {_CLOTH_ANALYSIS_CACHE[thumb_hash].category}")
        return _CLOTH_ANALYSIS_CACHE[thumb_hash]

    # 2. Check pre-registered catalog
    if prefix_16 in _PRE_REGISTERED_CATALOG:
        analysis = _PRE_REGISTERED_CATALOG[prefix_16]
        _CLOTH_ANALYSIS_CACHE[thumb_hash] = analysis
        print(f"[CLOTH ANALYSIS] Catalog hit for garment {prefix_16} -> {analysis.garment_name}")
        return analysis

    # 3. Dynamic Gemini Vision Analysis
    try:
        active_key = get_gemini_api_key()
        if not client and active_key:
            client = _get_client(active_key)

        if client:
            print(f"[CLOTH ANALYSIS] Performing dynamic vision analysis for garment {prefix_16}...")
            analysis_prompt = (
                "Analyze the clothing item in this image with high precision for a virtual try-on system.\n"
                "Determine:\n"
                "1. category: e.g. blazer, suit_jacket, suit, coat, jacket, shirt, t_shirt, dress, hoodie, etc.\n"
                "2. garment_name: concise descriptive name\n"
                "3. fabric_type: material texture, weight, and thickness (e.g. structured heavyweight suiting tweed/wool with canvas and shoulder pads, lightweight cotton, soft jersey, etc.)\n"
                "4. structure_level: 'rigid_tailored' (blazers, suits, coats with shoulder pads), 'semi_structured', or 'soft_drape'\n"
                "5. is_outerwear: boolean (true for blazers, suits, coats, jackets worn over other clothes; false for base shirts, tees, dresses)\n"
                "6. sleeve_behavior: explicit instruction for sleeve drape on arms (e.g. for blazer/suit: clean smooth tailored cylindrical drape with zero horizontal accordion folds, wrinkles, ripples, or bunching)\n"
                "7. tryon_guidelines: specific rules to avoid shirt-on-shirt, preserve authentic fabric thickness, lapels, pockets, and silhouette.\n"
                "8. sleeve_length: where the sleeves end, one of: sleeveless, short (above the elbow), elbow, three_quarter (mid-forearm), full (to the wrist)\n"
                "9. top_length: where the top's hem ends, one of: cropped (above the waist, midriff showing), waist, hip, thigh, knee, below_knee\n"
                "10. includes_bottom: boolean, true when the image shows a garment that dresses the legs itself (an outfit set with its own pants, salwar, palazzo, dhoti, skirt or churidar; a dress, frock, gown, saree or jumpsuit), false when it is a top only (shirt, t-shirt, kurta or kurti alone, blazer, jacket, top)\n\n"
                "Return as a JSON object with keys: category, garment_name, fabric_type, structure_level, is_outerwear, sleeve_behavior, tryon_guidelines, sleeve_length, top_length, includes_bottom."
            )

            call_started = time.perf_counter()
            resp = client.models.generate_content(
                model=ANALYSIS_MODEL,
                contents=[analysis_prompt, dress_img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            record_usage("cloth_analysis", ANALYSIS_MODEL, resp, duration_s=time.perf_counter() - call_started)
            if resp and resp.text:
                data = json.loads(resp.text)
                analysis = ClothAnalysis(
                    category=str(data.get("category", "apparel")).lower().strip(),
                    garment_name=str(data.get("garment_name", "Apparel Garment")),
                    fabric_type=str(data.get("fabric_type", "Woven fabric")),
                    structure_level=str(data.get("structure_level", "semi_structured")),
                    is_outerwear=bool(data.get("is_outerwear", False)),
                    sleeve_behavior=str(data.get("sleeve_behavior", "Natural sleeve drape")),
                    tryon_guidelines=str(data.get("tryon_guidelines", "Fit naturally to frame")),
                    sleeve_length=normalize_length(data.get("sleeve_length"), SLEEVE_LENGTHS),
                    top_length=normalize_length(data.get("top_length"), TOP_LENGTHS),
                    includes_bottom=data.get("includes_bottom") if isinstance(data.get("includes_bottom"), bool) else None,
                )
                _CLOTH_ANALYSIS_CACHE[thumb_hash] = analysis
                print(f"[CLOTH ANALYSIS] Successfully analyzed: {analysis.category} ({analysis.garment_name})")
                return analysis
    except Exception as err:
        print(f"[CLOTH ANALYSIS WARNING] Vision analysis failed ({err}), falling back to heuristic...")

    # 4. Resilient Heuristic Fallback
    w, h = dress_img.size
    aspect = h / max(1, w)
    is_likely_blazer = aspect < 1.30
    fallback = ClothAnalysis(
        category="blazer" if is_likely_blazer else "apparel",
        garment_name="Tailored Blazer" if is_likely_blazer else "Apparel",
        fabric_type="Tailored suiting fabric with structured body" if is_likely_blazer else "Standard apparel fabric",
        structure_level="rigid_tailored" if is_likely_blazer else "semi_structured",
        is_outerwear=is_likely_blazer,
        sleeve_behavior="Smooth, clean, structured cylindrical sleeve drape with zero horizontal folding or ripples on arms." if is_likely_blazer else "Natural smooth sleeve hang.",
        tryon_guidelines="Preserve authentic fabric thickness and structured silhouette. Do not render as thin shirt-on-shirt. Keep sleeve drape clean and smooth."
    )
    _CLOTH_ANALYSIS_CACHE[thumb_hash] = fallback
    return fallback


# In-memory cache for user facial and physical appearance analysis keyed by thumbnail hash
_USER_APPEARANCE_CACHE: Dict[str, str] = {}
# The same analysis' structured answers that feed garment coverage (own sleeve length, wrist accessories)
_USER_APPEARANCE_DETAILS: Dict[str, Dict[str, Optional[str]]] = {}


# Bump when the analysis prompt changes (the model is part of the key too)
APPEARANCE_PROMPT_VERSION = "v1"
MAX_APPEARANCE_ENTRIES = 16


def _appearance_hash(user_img: Image.Image) -> str:
    digest = hashlib.sha256(f"{ANALYSIS_MODEL}|{APPEARANCE_PROMPT_VERSION}|".encode())
    digest.update(user_img.convert("RGB").resize((128, 128)).tobytes())
    return digest.hexdigest()


def clear_user_appearance() -> None:
    """Visitor descriptions are personal data: dropped when a session is purged, and bounded in between."""
    _USER_APPEARANCE_CACHE.clear()
    _USER_APPEARANCE_DETAILS.clear()


def get_appearance_details(user_img: Image.Image) -> Dict[str, Optional[str]]:
    """Own sleeve length and wrist accessories from this capture's appearance analysis ({} if it has not run)."""
    return _USER_APPEARANCE_DETAILS.get(_appearance_hash(user_img), {})


def prepare_user_portrait(user_img: Image.Image) -> Image.Image:
    """
    Intelligently frames and crops wide webcam captures (e.g. 16:9, 1024x576)
    into a clean 3:4 portrait centered horizontally around the user's body.
    Increases facial and garment pixel density by 2.5x-4x, eliminates peripheral
    clutter (bystanders, office partitions), and matches the 3:4 output aspect ratio.
    """
    w, h = user_img.size
    aspect = w / float(h)
    # If already close to 3:4 portrait (0.75), return as is
    if 0.70 <= aspect <= 0.80:
        return user_img

    # If wider than 3:4 (e.g. 16:9 webcam ~1.78, 4:3 ~1.33, or 1:1 ~1.0)
    if aspect > 0.80:
        cx = w / 2.0
        try:
            from app.services.body_profile import body_profile_service
            lm = body_profile_service.extract_single_frame(user_img)
            if lm and lm.keypoints:
                xs = [pt[0] * w for pt in lm.keypoints.values() if pt is not None]
                if xs:
                    cx = (min(xs) + max(xs)) / 2.0
        except Exception:
            pass

        target_w = int(h * 0.75)
        if target_w <= w:
            left = int(cx - target_w / 2.0)
            left = max(0, min(w - target_w, left))
            right = left + target_w
            cropped = user_img.crop((left, 0, right, h))
            print(f"[PORTRAIT FRAMING] Cropped wide {w}x{h} frame -> centered 3:4 portrait ({target_w}x{h}).")
            return cropped
        else:
            target_h = int(w / 0.75)
            top = max(0, int((h - target_h) / 2.0))
            cropped = user_img.crop((0, top, w, min(h, top + target_h)))
            print(f"[PORTRAIT FRAMING] Cropped frame -> 3:4 portrait ({w}x{target_h}).")
            return cropped

    return user_img


def analyze_user_appearance(
    user_img: Image.Image,
    client: Optional[genai.Client] = None
) -> str:
    """
    Performs deep facial and physical identity analysis of the user for virtual try-on photographic replication.
    Extracts gender, ethnicity, exact skin tone & undertone, face shape, eyes, eyeglasses,
    hairstyle & drape, and photographic realism directives (anti-AI airbrushing, visible pores).
    Results are cached in-memory keyed by SHA-256 thumbnail hash for instant 0ms retrieval. A try-on arriving while the
    background preparation is analysing the same capture waits for that call instead of paying for a second one.
    """
    thumb_hash = _appearance_hash(user_img)
    with capture_cache.keyed_lock(f"appearance:{thumb_hash}"):
        return _analyze_user_appearance(user_img, thumb_hash, client)


def _analyze_user_appearance(user_img: Image.Image, thumb_hash: str, client: Optional[genai.Client]) -> str:
    if thumb_hash in _USER_APPEARANCE_CACHE:
        print(f"[USER APPEARANCE] Cache hit for user hash {thumb_hash[:16]}.")
        return _USER_APPEARANCE_CACHE[thumb_hash]

    active_key = get_gemini_api_key()
    if not client and active_key:
        client = _get_client(active_key)

    if client:
        try:
            print(f"[USER APPEARANCE] Analyzing user face and physical identity via Gemini Vision...")
            analysis_prompt = (
                "Analyze the person in this image for virtual try-on photographic identity replication.\n"
                "Return a JSON object with the following keys:\n"
                "- gender_presentation: concise gender / appearance (e.g. Female, Male)\n"
                "- ethnicity_heritage: ethnic or demographic heritage (e.g. South Asian, East Asian, Caucasian, etc.)\n"
                "- skin_tone: exact skin tone and undertone description (e.g. Medium warm-brown, wheatish, fair, dusky - NEVER lighten)\n"
                "- face_shape: face shape, chin, and jawline contour\n"
                "- eyes_and_eyebrows: eye shape/color, natural eyebrow shape and thickness\n"
                "- eyeglasses: exact glasses frame shape, color, rim style (e.g. dark rectangular-oval frames), or 'none'\n"
                "- hairstyle_and_drape: hair color, length, parting, and how hair falls over shoulders/chest\n"
                "- distinct_features: any distinctive facial markers or features\n"
                "- body_build: overall build exactly as photographed (e.g. lean, slim, average, stocky, heavyset), described plainly and neutrally\n"
                "- midsection: stomach and waist exactly as photographed (e.g. flat stomach, slight belly, rounded belly, prominent belly), described plainly and neutrally\n"
                "- own_sleeves: where the sleeves of the person's own top end, exactly as photographed: one of sleeveless, short (above the elbow), elbow (at the elbow or rolled up to it), three_quarter (mid-forearm), full (to the wrist)\n"
                "- own_bottom: what the person wears on the lower body, exactly as photographed: separate (trousers, jeans, pants, shorts, leggings or a skirt as a separate piece below a top), one_piece (a dress, frock, gown, jumpsuit or another single garment covering torso and legs), or not_visible (the lower body is out of the frame)\n"
                "- wrist_accessories: every watch, bracelet, bangle, kada or thread visible on each wrist, saying which side of the image that wrist is on (image left / image right), e.g. 'black wristwatch on the wrist at image right; red thread and steel kada on the wrist at image left' — or 'none'\n"
                "- realism_directives: explicit guidance for authentic photographic skin texture (visible pores, micro-texture, real lighting gradients) and strictly forbidding synthetic, AI-airbrushed, or plastic doll looks.\n"
            )

            call_started = time.perf_counter()
            resp = client.models.generate_content(
                model=ANALYSIS_MODEL,
                contents=[analysis_prompt, user_img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            record_usage("appearance_analysis", ANALYSIS_MODEL, resp, duration_s=time.perf_counter() - call_started)
            if resp and resp.text:
                data = json.loads(resp.text)
                gender = data.get("gender_presentation", "Individual")
                ethnicity = data.get("ethnicity_heritage", "")
                skin = data.get("skin_tone", "Natural skin tone matching Image 1")
                face_shape = data.get("face_shape", "Natural facial contours matching Image 1")
                eyes_brows = data.get("eyes_and_eyebrows", "Natural dark eyes and eyebrows")
                glasses = data.get("eyeglasses", "none")
                hair = data.get("hairstyle_and_drape", "Natural hair matching Image 1")
                build = data.get("body_build", "as photographed in Image 1")
                midsection = data.get("midsection", "as photographed in Image 1")
                realism = data.get(
                    "realism_directives",
                    "Render authentic human skin texture with natural pores, real lighting gradients, and true photographic fidelity. Strictly NO AI airbrushing or plastic smoothing."
                )

                formatted = (
                    f"\n[AUTHENTIC USER IDENTITY & PHOTOGRAPHIC REALISM — STRICTEST REQUIREMENT]\n"
                    f"- Demographics: {gender}{f', {ethnicity}' if ethnicity else ''}.\n"
                    f"- Skin Tone & Complexion: {skin}. Strictly do NOT lighten, bleach, or alter skin tone.\n"
                    f"- Face Shape & Structure: {face_shape}.\n"
                    f"- Eyes & Eyebrows: {eyes_brows}.\n"
                    f"- Eyeglasses: {glasses}.\n"
                    f"- Hairstyle & Drape: {hair}.\n"
                    f"- Body Build & Midsection (as photographed): {build}; {midsection}. Preserve exactly — do not slim or flatten.\n"
                    f"- Photographic Realism Directives: {realism}\n"
                    f"- ABSOLUTELY AVOID AI-GENERATED LOOK: Do NOT generate an airbrushed, plastic, doll-like, or synthetic CGI face. "
                    f"Render authentic human skin texture with subtle micro-texture and real indoor studio lighting. "
                    f"The face must look 100% like the exact real person in Image 1."
                )
                while len(_USER_APPEARANCE_CACHE) >= MAX_APPEARANCE_ENTRIES:
                    oldest = next(iter(_USER_APPEARANCE_CACHE))
                    del _USER_APPEARANCE_CACHE[oldest]
                    _USER_APPEARANCE_DETAILS.pop(oldest, None)
                _USER_APPEARANCE_CACHE[thumb_hash] = formatted
                wrist = str(data.get("wrist_accessories") or "").strip()
                _USER_APPEARANCE_DETAILS[thumb_hash] = {
                    "own_sleeves": normalize_length(data.get("own_sleeves"), SLEEVE_LENGTHS),
                    "wrist_accessories": None if wrist.lower() in ("", "none", "n/a", "no") else wrist,
                    "own_bottom": normalize_length(data.get("own_bottom"), OWN_BOTTOMS),
                }
                print(f"[USER APPEARANCE] Profile created and cached: {gender}, {skin}, glasses: {glasses}, build: {build}, midsection: {midsection}, "
                      f"own sleeves: {_USER_APPEARANCE_DETAILS[thumb_hash]['own_sleeves']}, wrists: {wrist or 'none'}")
                return formatted
        except Exception as err:
            print(f"[USER APPEARANCE WARNING] Vision analysis failed ({err}), falling back to standard identity lock.")

    # Not cached: a brief failure during background preparation must not cost this visitor the real analysis
    return (
        "\n[AUTHENTIC USER IDENTITY & PHOTOGRAPHIC REALISM — STRICTEST REQUIREMENT]\n"
        "- Exact facial identity lock: preserve the exact facial geometry, features, skin tone, glasses, and hair from Image 1.\n"
        "- Render authentic human skin texture with natural pores, subtle micro-texture, and real lighting.\n"
        "- Strictly NO AI airbrushing, plastic smoothing, doll-like porcelain skin, or synthetic CGI features."
    )


_TORSO_SHAPE_DESCRIPTIONS: Dict[str, str] = {
    "lean": "a lean, slim midsection with little volume at the stomach and waist",
    "average": "an average build with moderate volume at the stomach and waist",
    "full": "a fuller build with a visible stomach and a wide waist",
    "very_full": "a heavy build with a large, prominent stomach; the midsection is the fullest part of the torso",
}

_STRUCTURE_LABELS: Dict[str, str] = {
    "rigid_tailored": "rigid tailored (holds its own shape)",
    "semi_structured": "semi-structured",
    "soft_drape": "soft drape (follows the body)",
}


def build_tryon_prompt(
    fit_state: Optional[str] = None,
    body_profile: Optional[BodyProfile] = None,
    retry_strict: bool = False,
    is_inpainting_mode: Optional[bool] = None,
    cloth_analysis: Optional[ClothAnalysis] = None,
    user_appearance: Optional[str] = None,
    size_offset: Optional[object] = None,
    selected_size: Optional[str] = None,
    enforce_studio_background: bool = False,
    fit_escalation: Optional[str] = None,
    capture_coverage: Optional[CaptureCoverage] = None,
    coverage_escalation: Optional[str] = None,
    limb_escalation: Optional[str] = None,
) -> str:
    """
    Builds the generation prompt: base prompt (face and body lock), the visitor's appearance and measured body
    shape, the garment analysis and where the garment ends (see garment_coverage), the size fit (category, fabric
    physics and body-dependent drape, see fit_drape), and retry / studio directives. Every size is generated from the
    capture (Image 1) and the garment (Image 2) only; no earlier render is ever an input. fit_state and
    is_inpainting_mode are accepted for callers but no longer used: fit comes from size_offset.
    """
    prompt = PROMPT_TEXT
    torso_shape = body_profile.torso_shape if body_profile is not None else None

    if user_appearance:
        prompt += user_appearance.rstrip("\n") + "\n"

    if cloth_analysis is not None:
        structure = _STRUCTURE_LABELS.get(cloth_analysis.structure_level, cloth_analysis.structure_level)
        prompt += (
            f"\n[GARMENT — {cloth_analysis.garment_name} ({cloth_analysis.category})]\n"
            f"- Fabric: {cloth_analysis.fabric_type}\n"
            f"- Structure: {structure}{'; outerwear layered over the inner clothing' if cloth_analysis.is_outerwear else ''}\n"
            f"- Sleeves: {cloth_analysis.sleeve_behavior}\n"
            f"- Try-on guidelines: {cloth_analysis.tryon_guidelines}\n"
        )

    prompt += coverage_section(cloth_analysis, capture_coverage)

    if torso_shape in _TORSO_SHAPE_DESCRIPTIONS:
        prompt += (
            "\n[VISITOR BODY SHAPE — MEASURED FROM IMAGE 1, PRESERVE EXACTLY]\n"
            f"- This visitor has {_TORSO_SHAPE_DESCRIPTIONS[torso_shape]}.\n"
            "- Render exactly this body shape and volume under the garment. Do NOT slim, flatten, or idealize it, and do NOT add body volume that is not there (the extra fabric of a larger size is garment, not body).\n"
            "- Image 1 is the ground truth: if anything here disagrees with Image 1, follow Image 1.\n"
        )
        if size_offset is None:
            prompt += body_drape_text(torso_shape, "perfect", cloth_analysis)

    if size_offset is not None:
        prompt += size_fit_section(size_offset, torso_shape, cloth_analysis, selected_size)

    # Retry after a render whose size did not look different from another size's render
    if fit_escalation:
        prompt += fit_escalation

    # Retry after a render that showed the visitor's own sleeve past the garment's, or bare skin under its sleeve
    if coverage_escalation:
        prompt += coverage_escalation

    # Retry after a render that lost an arm, hand, leg or foot
    if limb_escalation:
        prompt += limb_escalation

    # 4. Retry Strict Constraint (if retry after silhouette mismatch)
    if retry_strict:
        prompt += (
            "\n[STRICT SILHOUETTE ALIGNMENT]\n"
            "- CRITICAL PRIORITY: the body under the garment must match Image 1 exactly: shoulder span, torso length, arm position, and the full width of the torso, stomach and waist.\n"
            "- Keep the torso and midsection exactly as wide and full as in Image 1: do not slim the waist or flatten the stomach.\n"
            "- Avoid any warping, limb drift, or boundary shifts.\n"
        )
        if getattr(size_offset, "fit_quality", "perfect") != "perfect":
            prompt += "- This locks the BODY only: the garment still fits as the size fit section says (a larger size hangs away from this body, a smaller size strains against it).\n"

    # 5. Studio Background Enforcement (retry after the model returned the raw capture scene)
    if enforce_studio_background:
        prompt += (
            "\n[MANDATORY — FULL STUDIO RE-RENDER, NOT A PHOTO EDIT]\n"
            "- A previous attempt returned the ORIGINAL photo with only the clothes swapped. That output is REJECTED.\n"
            "- Do NOT keep ANY part of Image 1's room: no walls, windows, decorations, furniture, bags, or office objects.\n"
            "- Place the person on a seamless, plain, evenly lit pure white / very light grey studio backdrop filling the entire frame edge to edge.\n"
            "- Re-light the person with soft, balanced professional studio lighting and a subtle floor contact shadow.\n"
        )

    return prompt


SIZE_EDIT_PROMPT = (
    "Edit the attached try-on image. Treat it as a fixed camera shot.\n"
    "Image 1 = current try-on result. Image 2 = original product photo of the {garment_type}.\n\n"
    "ALLOWED TO CHANGE: only the fit and drape of the {garment_type}\n"
    "- Re-drape it as a size {new_size} (was {old_size}): {fit_description}\n"
    "- Folds, wrinkles, how the fabric hangs, hem length and sleeve length may change naturally to match the new size.\n\n"
    "GARMENT DESIGN IS LOCKED (copy exactly from Image 2)\n"
    "- Same pattern, print, motifs and their scale. Do not redraw, simplify, enlarge, shrink or rearrange the pattern.\n"
    "- Same colours and shades, same fabric texture.\n"
    "- Same neckline, sleeve style, buttons, embroidery, borders, logos and their positions.\n"
    "- When the fabric gets looser, the pattern follows the folds as the same printed cloth; it is not regenerated or "
    "replaced with a new design.\n\n"
    "CAMERA AND PERSON ARE LOCKED (identical to Image 1)\n"
    "- Same zoom, crop, framing, field of view and angle.\n"
    "- Head top and feet stay at the same pixel positions; the person fills the same height of the frame.\n"
    "- Same pose, body size, proportions and face.\n"
    "- Background, lighting and output resolution ({width}x{height}) unchanged.\n\n"
    "A bigger size does NOT mean a wider shot or a new design. Never zoom out to fit looser clothing, never zoom in on "
    "the garment, and never alter the pattern. If extra fabric would extend past the frame edge, let it be cropped.\n"
)


def build_size_edit_prompt(
    base_render: Image.Image,
    cloth_analysis: ClothAnalysis,
    size_offset: object,
    selected_size: Optional[str],
    body_profile: Optional[BodyProfile] = None,
    fit_escalation: Optional[str] = None,
    limb_escalation: Optional[str] = None,
) -> str:
    """Prompt for re-draping the recommended-size render (Image 1) at another size; the fit text is the size-fit section."""
    torso_shape = body_profile.torso_shape if body_profile is not None else None
    prompt = SIZE_EDIT_PROMPT.format(
        garment_type=cloth_analysis.garment_name or cloth_analysis.category,
        new_size=(selected_size or "").upper() or "SELECTED",
        old_size=(getattr(size_offset, "recommended_size", None) or "").upper() or "recommended",
        fit_description=size_fit_section(size_offset, torso_shape, cloth_analysis, selected_size).strip(),
        width=base_render.width,
        height=base_render.height,
    )
    return prompt + (fit_escalation or "") + (limb_escalation or "")


def generate_tryon_image(
    user_img: Image.Image,
    dress_img: Image.Image,
    fit_state: Optional[str] = None,
    body_profile: Optional[BodyProfile] = None,
    retry_strict: bool = False,
    feathered_mask: Optional[np.ndarray] = None,
    cloth_analysis: Optional[ClothAnalysis] = None,
    user_appearance: Optional[str] = None,
    session_id: Optional[str] = None,
    size_offset: Optional[object] = None,
    selected_size: Optional[str] = None,
    enforce_studio_background: bool = False,
    attempt: Optional[int] = None,
    fit_escalation: Optional[str] = None,
    identity_face: Optional[Image.Image] = None,
    capture_coverage: Optional[CaptureCoverage] = None,
    coverage_escalation: Optional[str] = None,
    limb_escalation: Optional[str] = None,
    base_render: Optional[Image.Image] = None,
) -> Tuple[bytes, float, ClothAnalysis]:
    """Calls Gemini to try on the garment and returns (image_bytes, face_similarity, cloth_analysis).
    Auto-crops wide webcam captures to a 3:4 portrait centered on the user. The model only ever sees the capture and
    the garment: every size is anchored to the capture, never to an earlier render.

    identity_face: this visitor's first served render. Used only when the capture's face cannot be pasted cleanly
    (low_detail_skip: too small to enlarge, or its colour cannot be matched to the render): its face is composited onto this render after generation so every size shows one identical
    face. It is never sent to the model.

    capture_coverage: the capture's own sleeve length, wrist accessories and skin colour (garment_coverage.read_capture)
    for the prompt's coverage section; coverage_escalation: the retry directive after a failed coverage check.
    """
    active_key = get_gemini_api_key()
    if not active_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")

    # SECURE PRINT CORE: Mask the key so it only exposes the first 6 characters
    masked_key = active_key[:6] + "..." + active_key[-4:] if len(active_key) > 10 else "Invalid/Short Key"
    print(f"\n[API CONFIG] Using GEMINI_API_KEY: {masked_key} | Size: {selected_size or 'auto'}\n")

    client = _get_client(active_key)

    print("[API DEBUG] Gemini SDK ready:", {
        "api_key_present": bool(client._api_client.api_key),
        "vertexai": client._api_client.vertexai,
        "resolved_base_url": getattr(client._api_client._http_options, 'base_url', None),
    })

    # 1. 3:4 Portrait Framing: Automatically center the user and crop away side clutter/bystanders
    user_portrait = prepare_user_portrait(user_img)

    # Deterministic integer seed derivation: locks identity while offsetting per size selection
    # so diffusion noise field resolves distinct fabric drape and tailoring while remaining 100% deterministic per size
    seed_base = session_id if session_id else hashlib.sha256(user_portrait.tobytes()).hexdigest()
    base_seed_int = int(hashlib.sha256(seed_base.encode("utf-8")).hexdigest()[:8], 16)
    size_seed_offset = 0
    if size_offset is not None:
        size_steps = getattr(size_offset, 'steps_from_recommended', 0)
        size_seed_offset = size_steps * 303
    elif selected_size:
        from app.services.size_chart import SIZE_INDEX
        size_seed_offset = (SIZE_INDEX.get(selected_size.upper(), 2) - 2) * 303
    elif fit_state:
        fit_offsets = {"standard": 0, "fitted": 101, "relaxed": 202}
        size_seed_offset = fit_offsets.get(fit_state.lower().strip(), 0)
    # A retry must not replay the rejected attempt's noise field
    retry_salt = 7919 * max(0, (attempt or 1) - 1)
    stable_seed = (base_seed_int + size_seed_offset + retry_salt) % 2147483647

    # 2. Dynamic Garment Cloth Analysis
    if cloth_analysis is None:
        cloth_analysis = analyze_cloth(dress_img, client=client)

    # 3. Dynamic User Identity & Facial Appearance Analysis
    if user_appearance is None:
        user_appearance = analyze_user_appearance(user_portrait, client=client)

    print(f"[API CALL] Setup complete: Cloth={cloth_analysis.category} ('{cloth_analysis.garment_name}'), User Profile conditioned. Sending to Gemini...")

    final_prompt = build_tryon_prompt(
        fit_state=fit_state,
        body_profile=body_profile,
        retry_strict=retry_strict,
        is_inpainting_mode=(feathered_mask is not None),
        cloth_analysis=cloth_analysis,
        user_appearance=user_appearance,
        size_offset=size_offset,
        selected_size=selected_size,
        enforce_studio_background=enforce_studio_background,
        fit_escalation=fit_escalation,
        capture_coverage=capture_coverage,
        coverage_escalation=coverage_escalation,
        limb_escalation=limb_escalation,
    )

    if base_render is not None and size_offset is not None:
        # Size change: edit the recommended-size render so camera, person and pattern stay locked to it
        final_prompt = build_size_edit_prompt(base_render, cloth_analysis, size_offset, selected_size, body_profile,
                                              fit_escalation, limb_escalation)
        contents = [final_prompt, base_render, dress_img]
    else:
        # First render: capture + garment only
        contents = [final_prompt, user_portrait, dress_img]
    print(f"[API CALL] Generating from capture + garment only (seed={stable_seed}) | "
          f"fit {getattr(size_offset, 'fit_quality', 'n/a')} offset {getattr(size_offset, 'steps_from_recommended', 'n/a')}.")

    request_timing.lap("gemini_request")
    call_started = time.perf_counter()
    with request_timing.step("gemini_image"):
        response = client.models.generate_content(
            model=TRYON_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    image_size=TRYON_IMAGE_SIZE,
                    aspect_ratio="3:4",   # pick what fits a person shot
                ),
                seed=stable_seed,
            ),
        )

    record_usage("tryon_generation", TRYON_MODEL, response, session_id=session_id, size=selected_size, attempt=attempt,
                 fit_category=getattr(size_offset, "fit_quality", None),
                 size_offset=getattr(size_offset, "steps_from_recommended", None),
                 duration_s=time.perf_counter() - call_started)

    sdk_http = getattr(response, 'sdk_http_response', None)
    sdk_http_body = getattr(sdk_http, 'body', None)
    raw_http_payload = sdk_http_body

    prompt_feedback = getattr(response, 'prompt_feedback', None)
    print(
        "[API DEBUG] Gemini response summary:",
        {
            "candidate_count": len(response.candidates) if getattr(response, 'candidates', None) else 0,
            "parts_present": bool(getattr(response, 'parts', None)),
            "prompt_feedback": prompt_feedback,
            "sdk_http_present": sdk_http is not None,
            "sdk_http_body_present": bool(sdk_http_body),
            "sdk_http_body_type": type(sdk_http_body).__name__ if sdk_http is not None else None,
            "sdk_http_response_stream_present": bool(getattr(sdk_http, 'response_stream', None)),
        },
    )

    if prompt_feedback is not None and getattr(prompt_feedback, 'block_reason', None) is not None:
        print(
            "[API DEBUG] Gemini content blocked:",
            {
                "block_reason": prompt_feedback.block_reason,
                "block_reason_message": getattr(prompt_feedback, 'block_reason_message', None),
                "safety_ratings": getattr(prompt_feedback, 'safety_ratings', None),
            },
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"Gemini blocked the request: {prompt_feedback.block_reason}. "
                f"{getattr(prompt_feedback, 'block_reason_message', '')}"
            ),
        )

    generated_image_bytes = None
    response_parts = getattr(response, 'parts', None)
    if not response_parts and getattr(response, 'candidates', None) and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if getattr(candidate, 'content', None) and getattr(candidate.content, 'parts', None):
            response_parts = candidate.content.parts

    if not response_parts:
        candidate_count = len(response.candidates) if getattr(response, 'candidates', None) else 0
        parsed_body = raw_http_payload
        if isinstance(parsed_body, str) and parsed_body.strip().startswith(('{', '[')):
            try:
                parsed_body = json.loads(parsed_body)
            except Exception:
                pass
        response_dump = None
        try:
            response_dump = response.model_dump(include_none=False, exclude={'sdk_http_response'})
        except Exception:
            response_dump = None
        print(
            "[API DEBUG] Gemini failed parse:",
            {
                "candidate_count": candidate_count,
                "sdk_http_headers": getattr(sdk_http, 'headers', None),
                "sdk_http_body": parsed_body,
                "sdk_http_response_stream_type": type(getattr(sdk_http, 'response_stream', None)).__name__ if sdk_http is not None else None,
                "gemini_response_dump": response_dump,
            },
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini API returned no response parts. "
                f"Candidate count: {candidate_count}."
            ),
        )

    for part in response_parts:
        if getattr(part, 'text', None):
            print(f"[API DEBUG] Gemini text part: {part.text[:200]}")
        elif getattr(part, 'inline_data', None) and getattr(part.inline_data, 'data', None):
            generated_image_bytes = part.inline_data.data
            break

    if not generated_image_bytes:
        print(
            "[API DEBUG] Gemini response parts present but no image bytes found.",
            {"response_parts_count": len(response_parts), "response_part_types": [type(part).__name__ for part in response_parts]},
        )
        raise HTTPException(
            status_code=500,
            detail="Gemini API executed successfully but returned empty image byte streams.",
        )

    # MANDATORY: the model regenerates the face regardless of prompt wording, so the real
    # captured face is restored at full output resolution and independently verified.
    # Always from the capture (user_portrait) — never from a reference render.
    # Do not remove: tests/test_face_preservation.py guards this call and the real score.
    with request_timing.step("face_restore"):
        final_bytes, face_check = restore_original_face(user_portrait, generated_image_bytes, feathered_mask)
    if face_check.low_detail_skip:
        # Capture face cannot be pasted cleanly: too small (a blurry patch) or off-colour (face_check.reason). Every size
        # must still show the same face, so the visitor's first served render supplies it; without one, this render's
        # own single, clean face is served untouched.
        if identity_face is not None:
            with request_timing.step("face_restore"):
                locked_bytes, lock_check = restore_original_face(identity_face, generated_image_bytes)
            if lock_check.restored:
                print(f"[FACE LOCK] Capture face not pasteable ({face_check.reason}); composited the visitor's first-render face "
                      f"(match {lock_check.score:.3f}) so every size shows the same face.")
                return locked_bytes, lock_check.score, cloth_analysis
            print(f"[FACE LOCK] Could not composite the first-render face ({lock_check.reason}); "
                  f"serving this render's own face | {lock_check.details}")
        print(f"[FACE PRESERVE] {face_check.reason}; serving render uncomposited | {face_check.details}")
    elif not face_check.restored:
        # The model's own face must never be served: fail this attempt so the caller regenerates
        # (the /api/tryon attempt loop retries non-403 HTTP errors).
        print(f"[FACE PRESERVE] Render rejected: {face_check.reason} | {face_check.details}")
        raise HTTPException(status_code=502, detail=FACE_NOT_RESTORED_MESSAGE)
    return final_bytes, face_check.score, cloth_analysis
