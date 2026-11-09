"""
Size-fit section of the try-on prompt: how the selected size drapes on this visitor.

Four parts, combined with the prompt's body-shape and garment sections:
1. The fit category (from size_chart.get_size_offset), with slack or strain that scales with the size offset.
2. Fabric physics for the garment's structure (stiff tailoring, crisp woven, fluid drape, stretch knit).
3. Body-dependent drape: the same size reads differently on a lean and a fuller torso.
4. Where the strain shows on this body (smaller sizes): the widest part of this visitor, not a generic location.
"""
from typing import Dict, List, Optional

from app.models import ClothAnalysis
from app.services.size_chart import LOOSE_FITS, TIGHT_FITS

_SIZE_WORDS = {1: "ONE SIZE", 2: "TWO SIZES", 3: "THREE SIZES", 4: "FOUR SIZES", 5: "FIVE SIZES"}


def _sizes(n: int) -> str:
    return _SIZE_WORDS.get(abs(n), f"{abs(n)} SIZES")


# Every size other than the correct one: the size changes the garment's dimensions, never its design. Without this an
# image model reads "larger" as "more fabric" and adds frills, tiers or new drapes.
DESIGN_LOCK = (
    "- DESIGN LOCK — ONLY THE DIMENSIONS CHANGE: this size fit overrides any fit wording in the garment notes above, "
    "but it is the identical garment from Image 2: the same cut, panels, neckline, colours, print, embroidery and "
    "borders, in the same places, scaled to this size.\n"
    "- NEVER ADD DESIGN ELEMENTS FOR A SIZE: no frills, ruffles, flounces, tiers, pleats, gathers, smocking, peplum, "
    "extra layers or extra fabric pieces, and no more flare than the design has. A dupatta, stole, pallu or scarf is "
    "draped as in Image 2, except that it never covers a hand or forearm. Extra size shows only as width, longer sleeves and hem, lower shoulder seams and "
    "straight soft folds; a smaller size shows only as tension, shorter sleeves and hem, and strain lines.\n"
)


def _category_block(quality: str, steps: int, size: str, rec: str) -> str:
    if quality == "perfect":
        return (
            f"\n[SIZE FIT — SIZE {size}: CORRECT FIT (the visitor's recommended size)]\n"
            "- The garment fits this body as designed: shoulder seams sit on the shoulder points, long sleeves end at the "
            "wrist, the hem is at its designed length, and closures lie flat and closed with no pulling.\n"
            "- Moderate, even ease: the fabric follows the body with a small comfortable gap and a few soft natural "
            "folds; neither tight nor baggy.\n"
        )
    # Larger sizes, zone by zone: an image model renders concrete cues (sleeve length) readily and abstract ones
    # (a circumference in cm) weakly, so every zone gets a visible cue or the extra size lands only in the sleeves
    if quality == "relaxed":
        return (
            f"\n[SIZE FIT — SIZE {size}: RELAXED FIT, {_sizes(steps)} LARGER THAN THE VISITOR'S SIZE {rec}]\n"
            "- The WHOLE garment is one size bigger, the body of the garment as much as the sleeves: the torso looks as "
            "roomy as the sleeves do.\n"
            "- SHOULDERS: the shoulder seams drop about 2–3 cm past the shoulder points onto the upper arm, so the "
            "garment's shoulder line is visibly wider than the body's.\n"
            "- CHEST: about 8–10 cm more chest circumference than this body needs. The fabric no longer touches the sides "
            "of the chest: a soft vertical fold runs down from each armpit and the front lies slightly off the chest.\n"
            "- WAIST AND SIDES: the body of the garment falls straight down from the chest instead of following the "
            "waist, with a visible gap between the fabric and the sides of the body and a few soft vertical folds to the "
            "hem; a tucked or belted hem simply sits a little looser at the waist.\n"
            "- SLEEVES: a little wider around the arm; long sleeves about 2 cm too long, reaching the base of the thumb "
            "(short sleeves sit a little lower and stand slightly off the arm).\n"
            "- HEM: about 3 cm longer.\n"
            "- Clearly roomier than the correct size, but still neat.\n"
        )
    if quality == "loose":
        return (
            f"\n[SIZE FIT — SIZE {size}: CLEARLY LOOSE, {_sizes(steps)} LARGER THAN THE VISITOR'S SIZE {rec}]\n"
            "- The WHOLE garment is clearly too big, the body of the garment as much as the sleeves.\n"
            "- SHOULDERS: the shoulder seams drop 5–7 cm down onto the upper arm and the armholes sit low; the shoulder "
            "line slopes and is clearly wider than the body's.\n"
            "- CHEST: about 15–20 cm more chest circumference than this body needs: the front and sides stand clearly away "
            "from the chest, with deep vertical folds running down from each armpit.\n"
            "- WAIST AND SIDES: the body of the garment hangs straight and wide like a box, well outside the body's "
            "outline at the sides and back, in long straight vertical folds to the hem.\n"
            "- SLEEVES: wide and long: long sleeves 4–5 cm too long, covering the wrist bone with a few soft folds above "
            "the cuff; both hands stay fully visible below the cuffs (short sleeves reach toward the elbow).\n"
            "- HEM: 5–7 cm longer, well down over the hips.\n"
        )
    if quality == "oversized":
        return (
            f"\n[SIZE FIT — SIZE {size}: CLEARLY BAGGY, SAME DESIGN, {_sizes(steps)} LARGER THAN THE VISITOR'S SIZE {rec}]\n"
            f"- The WHOLE garment is clearly baggy: {'25' if steps <= 3 else '30'} cm or more extra chest circumference"
            f"{', even more than three sizes up would' if steps > 3 else ''}, in the body of the garment as much as in "
            "the sleeves.\n"
            "- SHOULDERS: the shoulder seams hang 8–12 cm down the upper arm; the armholes sit far below the armpits.\n"
            "- CHEST, WAIST AND SIDES: the body of the garment is a wider version of the same shape: it hangs straight "
            "down from the chest and shoulder blades in long, smooth vertical folds, well outside the body's outline at "
            "the sides; the body's outline shows only where the fabric rests on it.\n"
            "- SLEEVES: wide and straight; long sleeves reach the base of the thumb with a few soft folds above the cuff, "
            "and both hands stay fully visible below the cuffs (short sleeves reach the elbow).\n"
            "- HEM: 8–12 cm longer, covering the hips and seat.\n"
        )
    if quality == "snug":
        return (
            f"\n[SIZE FIT — SIZE {size}: SNUG, {_sizes(steps)} SMALLER THAN THE VISITOR'S SIZE {rec}]\n"
            "- Almost no ease: the fabric lies close against the body everywhere.\n"
            "- Light pulling across the widest part of the torso: fine horizontal tension lines there and, on a buttoned "
            "front, small gaps pulling open between the buttons at that point.\n"
            "- Sleeves fit closely around the upper arms and end 1–2 cm short (a long sleeve shows the wrist bone); the "
            "hem sits about 2 cm higher.\n"
            "- Shoulder seams sit on or just inside the shoulder points.\n"
            "- It closes and is wearable, but it is visibly on the small side.\n"
        )
    if quality == "tight":
        return (
            f"\n[SIZE FIT — SIZE {size}: TIGHT WITH VISIBLE STRAIN, {_sizes(steps)} SMALLER THAN THE VISITOR'S SIZE {rec}]\n"
            "- The garment is stretched over the body: strong horizontal strain lines across the chest, stomach and "
            "upper arms, radiating from the points where the body is widest.\n"
            "- Buttons and seams under clear tension: a buttoned placket pulls into X-shaped folds and gapes open between "
            "the buttons (the gaps show the visitor's own clothing from Image 1 or a plain inner layer, never bare skin); "
            "the side seams are pulled forward.\n"
            "- Shoulder seams sit 1–2 cm inside the shoulder points and the back pulls across the shoulder blades; the "
            "sleeves grip the upper arms and end 3–4 cm short, the cuff above the wrist bone.\n"
            "- The hem rides up 3–5 cm, to the top of the waistband.\n"
            "- It is still closed, but obviously too small.\n"
        )
    # does_not_fit
    return (
        f"\n[SIZE FIT — SIZE {size}: WON'T FIT — TOO SMALL FOR THIS BODY (the visitor's size is {rec})]\n"
        "Show honestly what happens when this visitor puts on a garment that is too small for them. It must look like a "
        "real photograph of a real garment that is too small, not an effect.\n"
        "- The garment cannot close: on a buttoned or zipped front only the top is fastened; the lower fastenings are "
        "open because the two edges no longer reach each other, leaving a clear gap over the widest part of the torso. "
        "The gap shows the visitor's own clothing from Image 1 or a plain inner layer, never bare skin. On a pull-over "
        "garment the fabric is stretched to its limit and rides up to the top of the waistband.\n"
        "- The fabric is stretched thin and pulled into hard horizontal ridges; every seam is visibly straining.\n"
        "- Shoulder seams sit well inside the shoulder points and the back is pulled tight across the shoulder blades; "
        "the sleeves are tight bands around the upper arms, pulled up so a long sleeve ends several centimetres above "
        "the wrist, and the armholes visibly restrict the arms (the arms stay in the pose of Image 1).\n"
        "- The hem rides well above its designed length.\n"
        "- Physically plausible only: the garment is worn ON the body and touches it everywhere it is stretched. No "
        "floating fabric, no fabric passing through the body, no torn or missing panels, and the body is NOT squeezed "
        "smaller: the garment strains around the body's real size.\n"
    )


def _is_knit(cloth: ClothAnalysis) -> bool:
    text = f"{cloth.fabric_type} {cloth.category}".lower()
    return any(k in text for k in ("knit", "jersey", "stretch", "elastane", "spandex", "lycra", "t_shirt", "polo"))


# Fabric physics: stiff tailoring, crisp woven and fluid drapes fall and strain differently
_FABRIC_LOOSE: Dict[str, str] = {
    "rigid_tailored": "Structured tailoring does not collapse into soft folds: the extra size shows as padded shoulders "
                      "extending past the body's shoulders, a boxy body standing away from the torso, fronts and lapels "
                      "hanging straight and wide, and sleeves hanging past the wrist as stiff tubes with a break at the cuff.",
    "semi_structured": "Crisp woven fabric holds some body: the slack forms a few broad folds and soft vertical creases rather "
                       "than ripples, and the fabric stands a little off the body instead of clinging.",
    "soft_drape": "Soft, fluid fabric falls under its own weight: the slack hangs straight down in long, soft vertical "
                  "folds and clings lightly where it touches the body; it does not ripple, ruffle or pool.",
}
_FABRIC_TIGHT: Dict[str, str] = {
    "rigid_tailored": "Structured tailoring cannot stretch: the fastened front button pulls the fronts into X-shaped strain "
                      "folds, the lapels bow away from the chest, the back seam pulls across the shoulder blades, and the "
                      "sleeves ride up the forearms.",
    "semi_structured": "Woven fabric has little stretch: tension shows as sharp horizontal pull lines between the buttons and "
                       "diagonal drag lines toward the widest point, with the fabric smooth and taut where it is stretched.",
    "soft_drape": "Soft fabric stretches and clings: it outlines every contour of the torso and arms, thins visibly over "
                  "the widest point, and shows fine horizontal stretch creases.",
}
_KNIT_LOOSE = "It is a knit: the excess hangs heavier and straighter, in soft vertical folds, and the ribbed hem and cuffs sit loose."
_KNIT_TIGHT = ("It is a stretch knit: it stretches instead of gaping, so the strain shows as the fabric clinging, thinning "
               "and turning slightly paler over the widest point, with ribbed cuffs and hem pulled tight.")


def _fabric_text(cloth: Optional[ClothAnalysis], quality: str) -> str:
    if cloth is None or quality == "perfect":
        return ""
    structure = cloth.structure_level if cloth.structure_level in _FABRIC_LOOSE else "semi_structured"
    knit = _is_knit(cloth) and structure != "rigid_tailored"
    if quality in LOOSE_FITS:
        text = _FABRIC_LOOSE[structure] + (f" {_KNIT_LOOSE}" if knit else "")
    else:
        text = _FABRIC_TIGHT[structure] + (f" {_KNIT_TIGHT}" if knit else "")
    return f"- FABRIC BEHAVIOUR ({cloth.fabric_type}): {text}\n"


# How the same size reads on different bodies. A garment's dimensions are fixed for a size, so the visitor's own
# volume decides where it hangs slack and where it pulls: a lean body shows more slack in a larger size than a fuller
# one; a fuller body shows much more strain in a smaller size than a lean one, concentrated at the stomach. On a fuller
# body a larger size still shows on the torso (shoulders, chest, sides, below the stomach), never only in the sleeves:
# wording that played the torso down made larger sizes render as loose sleeves on a fitted body.
_BODY_DRAPE: Dict[str, Dict[str, str]] = {
    "lean": {
        "perfect": "Even at the correct size this lean body leaves visible ease: fabric falls straight from the chest with soft vertical folds and visible slack at the waist and sides. It does not cling to the stomach.",
        "relaxed": "On this lean body even one size up reads clearly: the fabric falls straight from the chest with obvious empty space at the waist and sides; nothing rests on the stomach.",
        "loose": "On this lean body the garment hangs away from the torso on all sides: empty fabric in deep vertical folds at the sides and back, a wide gap between the fabric and the waist, the hem hanging straight and wide.",
        "oversized": "On this lean body the garment is clearly baggy: it hangs straight and wide, well away from the torso on every side, and the body's outline shows only at the slope of the shoulders.",
        "snug": "On this lean body one size down is only mildly tight: the fabric lies smooth and close, with light tension across the shoulders and chest; over the flat stomach it lies flat without pulling.",
        "tight": "On this lean body the strain concentrates where the frame is widest, across the shoulders, chest and upper arms: horizontal pull lines there and sleeves gripping the arms, while the stomach area lies smooth and taut rather than pulled.",
        "does_not_fit": "On this lean body the garment fails at the frame: the fronts cannot meet across the chest, the back pulls tight across the shoulder blades, and the sleeves bind the upper arms; the flat stomach itself is not what stops it.",
    },
    "average": {
        "perfect": "Fabric follows the torso with moderate ease and soft folds at the waist.",
        "relaxed": "The extra room shows at the sides and back in soft vertical folds; the fabric rests lightly on the chest.",
        "loose": "The garment hangs boxy and wide around the torso, with long straight folds at the sides.",
        "oversized": "The garment hangs far off the torso in a large, boxy shape with long straight folds; the torso shows only where the fabric rests on the chest.",
        "snug": "Light tension spreads across the chest and stomach with fine horizontal pull lines.",
        "tight": "Strain spreads across the chest and stomach, with horizontal pull lines at both and the closure pulling between fastenings over the chest.",
        "does_not_fit": "The garment cannot close over the chest and stomach; hard strain lines run across both and the sleeves bind the upper arms.",
    },
    "full": {
        "perfect": "Because this visitor has a fuller midsection, the garment follows their real contours: fabric rests on the stomach, drapes over its widest point and falls from there, with gentle tension lines where it is widest. Do not smooth the stomach away.",
        "relaxed": "On this fuller body one size up still shows on the torso, not only at the sleeves: slack at the shoulders, chest and sides, with the sides of the garment standing away from the body. Over the stomach the fabric rests on the belly without any pull lines and falls straight down from its widest point, standing away from the body below it.",
        "loose": "Clearly too big on the torso as well as the sleeves: deep slack at the shoulders, chest and sides, the sides of the garment hanging well outside the body. Over the stomach the fabric rests on the belly without tension and hangs straight down from its widest point, well away from the body below it, with excess fabric at the sides.",
        "oversized": "The garment is clearly oversized at the shoulders, chest, sides and sleeves; over the stomach it rests on the belly and falls straight from its widest point, with large excess fabric at the sides and back.",
        "snug": "Clear pull at the widest point of the stomach, noticeably more strain than a lean body shows at the same size: horizontal strain lines across the belly, the front closure pulling between fastenings, and the hem riding up at the front.",
        "tight": "Heavy strain at the stomach, the widest part of this body: fabric stretched tight over the belly with pronounced horizontal pull lines, buttons straining and gaping across the stomach, side seams pulled forward, and the front hem riding up to the waistband.",
        "does_not_fit": "The stomach is where it fails: the fronts cannot close over the belly and the lower fastenings stand open with a clear gap at the stomach; above it the fabric is pulled into hard ridges and the front hem rides far up.",
    },
    "very_full": {
        "perfect": "The stomach carries the garment: fabric drapes over the widest point of the stomach with light pulling and tension lines there, the front closure follows the curve of the belly, and the front hem sits slightly higher than the back. Do not smooth or flatten the stomach.",
        "relaxed": "One size up still shows on the torso, not only at the sleeves: roomy shoulders and chest, and slack at the sides above and beside the stomach. Over the stomach the fabric rests on the belly with no strain at all (no pull lines, the closure lies flat) and falls straight down from its widest point, standing away from the body below it.",
        "loose": "Clearly too big on the torso as well as the sleeves: slack shoulders and chest, the sides of the garment hanging outside the body. Over the stomach the fabric rests on the belly with no strain and falls straight down from its widest point, well away from the body below it.",
        "oversized": "The shoulders, chest, sides and sleeves are clearly oversized; over the stomach the fabric rests on the belly without strain and falls straight, with large excess fabric at the sides and back.",
        "snug": "Strong strain at the widest point of the stomach, far more than on a lean body at the same size: pronounced horizontal pull lines, the front closure gaping between fastenings, fabric stretched tight over the belly, and the hem riding up at the front.",
        "tight": "Severe strain over the stomach: the fabric is stretched to its limit across the belly with deep horizontal ridges, the front closure gapes wide between fastenings, the side seams are dragged forward, and the front hem rides up well above the waistband.",
        "does_not_fit": "The stomach is far too large for this size: the fronts stop well short of each other over the belly, the lower fastenings cannot be closed and leave a wide gap, and the fabric above is stretched into hard ridges while the front hem rides far up.",
    },
}

# Structured garments hold their own shape, so body volume shows at the closure rather than as fabric wrapping the body
_RIGID_DRAPE_NOTE: Dict[str, str] = {
    "lean": "This is a structured tailored garment: on this lean body it stands slightly away from the torso at the waist, with visible space between the garment and the body.",
    "full": "This is a structured tailored garment: it does not cling. The fuller midsection shows as the fronts spreading over the stomach and the front button pulling at the waist.",
    "very_full": "This is a structured tailored garment: it does not cling. The stomach pushes the fronts apart, the front button pulls visibly at the waist, and the lapels bow outward.",
}

# Outfit sets with draped, unsized parts (a dhoti, a gathered salwar, a dupatta): they look the same at every size and
# only the fitted top follows the size fit
_FIXED_DRAPE: Dict[str, str] = {
    "sherwani": "The ivory dhoti is draped, not sized, so it looks the same at every size: soft fabric falling from the "
                "waist in deep, loose cowl folds between the legs and gathered at the ankles. It shows no strain and no "
                "extra slack, and does not follow the exact shape of the legs. Only the sherwani jacket follows the size "
                "fit above: its shoulders, chest, sleeves and hem.",
    "kurta_dhoti": "The white dhoti is wrapped and tied, not sized, so it looks the same at every size: tied at the "
                   "waist in the traditional front-pleated style, with neat vertical pleats at the centre front falling "
                   "to the ankles and the maroon border along its edge and hem. It shows no strain and no extra slack, "
                   "and does not follow the exact shape of the legs. Only the kurta follows the size fit above: its "
                   "shoulders, chest, sleeves and hem.",
    "salwar_suit": "The salwar and the dupatta are not sized, so they look the same at every size. The salwar is loose "
                   "by design: gathered at the waist, falling in soft relaxed folds down the legs to the embroidered "
                   "ankle border, with no strain and no change in fullness between sizes. The dupatta is a draped "
                   "scarf: it hangs over the left shoulder in the same way at every size, in proportion to the kameez, "
                   "and never shows tightness or slack. Only the kameez follows the size fit above: its shoulders, "
                   "chest, sleeves and hem.",
    "lehenga": "The lehenga skirt and the dupatta are not sized, so they look the same at every size. The skirt is "
               "flared by design: gathered at the waist and spreading in wide panels to the floor, with the same "
               "fullness, length and hem border at every size. The dupatta is draped diagonally across the body from "
               "the left shoulder to the right hip in the same way at every size and never shows tightness or slack. "
               "Only the fitted blouse follows the size fit above: its shoulders, bust, short sleeves and hem.",
    "royal_sherwani": "The embroidered stole and the maroon turban are accessories, not sized, so they look the same at "
                      "every size: the stole is draped around the neck and hangs down the front over both shoulders, "
                      "and the turban sits on the head above the hairline. Only the sherwani, the inner kurta and the "
                      "churidar follow the size fit.",
    "saree": "The saree and the necklace are not sized, so they look the same at every size: the cream saree is "
             "wrapped at the waist with neat vertical pleats at the centre front falling straight to the floor, and "
             "the gold-bordered pallu goes diagonally across the chest and over the left shoulder, draped the same "
             "way at every size; the gold necklace rests at the neckline. Only the fitted green blouse follows the size "
             "fit above: its shoulders, bust, short sleeves and hem.",
    "mundu_shirt": "The mundu and the shawl are not sized, so they look the same at every size: the mundu is "
                   "wrapped at the waist and falls straight to the ankles with its gold border down the front overlap "
                   "and along the hem, and the shawl hangs over the left shoulder down the front to the thigh. Only the "
                   "shirt follows the size fit above: its shoulders, chest and sleeves; it stays tucked into the mundu.",
}

# Outfit sets whose fitted pants are sized with the top: the pants show the same fit category, below the top's hem
_SIZED_PANTS: Dict[str, Dict[str, str]] = {
    "kurta_pajama": {
        "perfect": "The pants fit as designed too: neat and straight below the kurta hem, following the legs with a "
                   "little ease, ending at the ankle bone with a clean hem.",
        "relaxed": "The pants are one size roomy too: a little looser below the kurta hem and about 2 cm longer, with a "
                   "soft break resting on the foot.",
        "loose": "The pants are clearly too big too: wide and slack below the kurta hem, 4–5 cm too long, bunching in "
                 "folds over the ankles and the top of the feet.",
        "oversized": "The pants are far too big too: baggy tubes below the kurta hem, about 6–8 cm too long, resting in "
                     "a deep break on the top of the feet; both feet stay visible.",
        "snug": "The pants are a little small too: close against the calves below the kurta hem, ending 1–2 cm above "
                "the ankle bone.",
        "tight": "The pants are tight too: stretched against the calves below the kurta hem with fine horizontal pull "
                 "lines, the hem 3–4 cm above the ankle so the ankles show.",
        "does_not_fit": "The pants are too small too: stretched hard against the legs with horizontal strain lines, "
                        "the hem riding well above the ankles; the legs are not squeezed thinner.",
    },
    # A churidar is cut long and gathers at the ankle by design: its size shows in how close it fits the calf and how
    # much excess gathers at the ankle
    "royal_sherwani": {
        "perfect": "The churidar fits as designed too: close-fitting down the calves below the sherwani hem and "
                   "gathered in soft bunched folds at the ankles.",
        "relaxed": "The churidar is one size roomy too: a little looser down the calves, with deeper, fuller gathers "
                   "bunching at the ankles.",
        "loose": "The churidar is clearly too big too: loose down the calves, with deeper gathers at the ankles; both "
                 "feet stay visible.",
        "oversized": "The churidar is far too big too: baggy down the legs, with the deepest gathers at the ankles; both "
                     "feet stay visible.",
        "snug": "The churidar is a little small too: tight down the calves, with only a few shallow gathers at the "
                "ankles.",
        "tight": "The churidar is tight too: stretched smooth against the calves with fine horizontal pull lines and "
                 "almost no gathers left, riding just above the ankle bone.",
        "does_not_fit": "The churidar is too small too: stretched hard against the legs with horizontal strain lines "
                        "and no gathers at all, the hem pulled above the ankles; the legs are not squeezed thinner.",
    },
}

_LIMIT_ZONES = {"chest": "chest and upper arms", "waist": "stomach and waist", "shoulder": "shoulders and upper back"}


def strain_zones(torso_shape: Optional[str], limiting_dimension: Optional[str]) -> List[str]:
    """Where a too-small garment shows tension on this body: its widest part first."""
    zones = []
    if torso_shape in ("full", "very_full"):
        zones.append("stomach")
    if limiting_dimension in _LIMIT_ZONES:
        zones.append(_LIMIT_ZONES[limiting_dimension])
    if not zones:
        zones.append("shoulders and chest" if torso_shape == "lean" else "chest and stomach")
    return list(dict.fromkeys(zones))


def body_drape_text(torso_shape: Optional[str], fit_quality: str, cloth_analysis: Optional[ClothAnalysis]) -> str:
    """Body-dependent drape line for a fit category; empty when the body shape was not measured."""
    drape = _BODY_DRAPE.get(torso_shape or "", {}).get(fit_quality)
    if not drape:
        return ""
    text = f"- BODY-DEPENDENT DRAPE: {drape}\n"
    if cloth_analysis is not None and cloth_analysis.structure_level == "rigid_tailored" and torso_shape in _RIGID_DRAPE_NOTE:
        text += f"- {_RIGID_DRAPE_NOTE[torso_shape]}\n"
    return text


def size_fit_section(
    size_offset: object,
    torso_shape: Optional[str],
    cloth_analysis: Optional[ClothAnalysis],
    selected_size: Optional[str] = None,
) -> str:
    """The full size-fit section for one generation."""
    quality = getattr(size_offset, "fit_quality", "perfect")
    steps = getattr(size_offset, "steps_from_recommended", 0)
    size = (selected_size or getattr(size_offset, "selected_size", None) or "").upper() or "SELECTED"
    rec = (getattr(size_offset, "recommended_size", None) or "").upper() or "(recommended)"
    lo_mm = getattr(size_offset, "offset_lo_mm", 15.0)
    hi_mm = getattr(size_offset, "offset_hi_mm", 25.0)

    text = _category_block(quality, steps, size, rec)
    text += f"- Garment-to-body offset: {lo_mm:.0f}mm to {hi_mm:.0f}mm (negative = the garment is smaller than the body).\n"
    text += _fabric_text(cloth_analysis, quality)
    text += body_drape_text(torso_shape, quality, cloth_analysis)
    if quality in LOOSE_FITS:
        text += ("- BODY vs GARMENT: only the BODY is frozen, not the garment's outline. The body underneath keeps exactly "
                 "the shape, width and stomach of Image 1, while this larger garment stands outside it at the shoulders, "
                 "chest, sides and hem, so the clothed torso looks wider and roomier than at the correct size. Loose "
                 "sleeves on a close-fitting torso are wrong: the torso is exactly as roomy as the sleeves.\n")
    if quality in TIGHT_FITS:
        zones = strain_zones(torso_shape, getattr(size_offset, "limiting_dimension", None))
        text += (f"- WHERE THE STRAIN SHOWS: this visitor's body is widest at the {' and the '.join(zones)}; the tension, "
                 "pull lines and any gaping concentrate there.\n")
    if quality != "perfect":
        text += DESIGN_LOCK
    category = (cloth_analysis.category or "").lower() if cloth_analysis is not None else ""
    sized_pants = _SIZED_PANTS.get(category, {}).get(quality)
    if sized_pants:
        text += f"- PANTS AT THIS SIZE: {sized_pants}\n"
    fixed_parts = _FIXED_DRAPE.get(category)
    if fixed_parts:
        text += f"- PARTS THAT DO NOT CHANGE WITH SIZE: {fixed_parts}\n"
    text += ("- 100% FROZEN PHYSICAL BODY: the body, stomach, proportions and pose are exactly as in Image 1 at every "
             "size; only the garment's fit and fabric behaviour change.\n")
    return text


def fit_escalation_text(fit_quality: str, other_size: Optional[str], problem: str) -> str:
    """Retry directive after a render that did not show its size (too close to another size's render)."""
    direction = ("LOOSER and LARGER: more slack, lower shoulder seams, longer sleeves and hem, wider silhouette"
                 if fit_quality in LOOSE_FITS else
                 "TIGHTER and SMALLER: more strain lines and pulling, shorter sleeves and hem riding up, fabric hugging the body")
    against = f" size {other_size}" if other_size else " the other size"
    return (
        "\n[SIZE DIFFERENCE NOT VISIBLE — REJECTED]\n"
        "- Show the size through the garment's dimensions only; its design stays exactly as in Image 2.\n"
        f"- The previous attempt at this size {problem}{against}. That output is rejected.\n"
        f"- Make the fit unmistakably {direction}, exactly as the size fit section describes, while the body stays identical.\n"
    )
