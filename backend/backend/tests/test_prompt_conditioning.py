import unittest
from app.models import BodyProfile, ClothAnalysis
from app.services import image_utils
from app.services.gemini_client import PROMPT_TEXT, build_tryon_prompt


def _profile(torso_shape=None, torso_fullness=None, body_type="Athletic / V-Shape"):
    return BodyProfile(
        height_ratio=0.82,
        shoulder_width_ratio=0.29,
        torso_aspect_ratio=1.12,
        shoulder_to_hip_ratio=1.25,
        chest_proportion=1.06,
        waist_proportion=0.83,
        hip_proportion=1.0,
        body_type=body_type,
        confidence_score=0.95,
        frame_count_sampled=3,
        torso_fullness=torso_fullness,
        torso_shape=torso_shape,
    )


def _garment(structure_level="semi_structured", category="shirt"):
    return ClothAnalysis(
        category=category,
        garment_name="Test Garment",
        fabric_type="Crisp woven cotton",
        structure_level=structure_level,
        is_outerwear=structure_level == "rigid_tailored",
        sleeve_behavior="Clean sleeve drape to the wrists",
        tryon_guidelines="Keep the placket straight",
    )


def _offset(steps, lo_mm, hi_mm, quality, dnf=False, rec="M", sel=None, limiting=None):
    """Create a mock SizeOffset-like object."""
    class _MockOffset:
        pass
    obj = _MockOffset()
    obj.steps_from_recommended = steps
    obj.offset_lo_mm = lo_mm
    obj.offset_hi_mm = hi_mm
    obj.fit_quality = quality
    obj.does_not_fit = dnf
    obj.recommended_size = rec
    obj.selected_size = sel
    obj.limiting_dimension = limiting
    return obj


PERFECT = (0, 15.0, 25.0, "perfect")
SNUG = (-1, 0.0, 10.0, "snug")
TIGHT = (-2, -10.0, 0.0, "tight")
WONT_FIT = (-2, -25.0, -10.0, "does_not_fit", True)
RELAXED = (1, 30.0, 45.0, "relaxed")
LOOSE = (2, 45.0, 60.0, "loose")
OVERSIZED = (3, 60.0, 80.0, "oversized")
ALL_FITS = (WONT_FIT, TIGHT, SNUG, PERFECT, RELAXED, LOOSE, OVERSIZED)


class TestPromptConditioning(unittest.TestCase):
    def test_backward_compatibility_fallback(self):
        """When no profile and no custom fit is provided, the prompt is the base prompt plus the lower-body rule."""
        prompt = build_tryon_prompt(fit_state=None, body_profile=None)
        self.assertTrue(prompt.startswith(PROMPT_TEXT))
        self.assertIn("LOWER BODY", prompt[len(PROMPT_TEXT):])

    def test_shape_preservation_always_present(self):
        """Every prompt (any garment, any size) forbids slimming or flattening the real body."""
        for cloth in (None, _garment("rigid_tailored", "blazer")):
            for offset in (None,) + tuple(_offset(*f) for f in ALL_FITS):
                prompt = build_tryon_prompt(cloth_analysis=cloth, size_offset=offset)
                self.assertIn("### 2. Body (locked)", prompt)
                self.assertIn("Never slim, enlarge, flatten, reshape or idealize the body", prompt)
                self.assertIn("waist, stomach and natural posture", prompt)

    def test_body_shape_block_from_measured_profile(self):
        """A measured torso shape is described to the model, with an explicit do-not-slim instruction."""
        prompt = build_tryon_prompt(body_profile=_profile("very_full", 1.25))
        self.assertIn("VISITOR BODY SHAPE — MEASURED FROM IMAGE 1", prompt)
        self.assertIn("large, prominent stomach", prompt)
        self.assertIn("Do NOT slim, flatten, or idealize it", prompt)
        self.assertIn("if anything here disagrees with Image 1, follow Image 1", prompt)

    def test_unmeasured_profile_adds_no_shape_block(self):
        """Without a torso measurement there is no shape description and no body-dependent drape line."""
        prompt = build_tryon_prompt(body_profile=_profile(), size_offset=_offset(*PERFECT))
        self.assertNotIn("VISITOR BODY SHAPE", prompt)
        self.assertNotIn("BODY-DEPENDENT DRAPE", prompt)
        self.assertIn("CORRECT FIT (the visitor's recommended size)", prompt)

    def test_lean_recommended_size_shows_slack(self):
        """At the correct size, a lean body leaves visible ease instead of fabric clinging to the stomach."""
        prompt = build_tryon_prompt(body_profile=_profile("lean", 0.9), size_offset=_offset(*PERFECT))
        self.assertIn("BODY-DEPENDENT DRAPE", prompt)
        self.assertIn("visible slack at the waist and sides", prompt)
        self.assertIn("It does not cling to the stomach", prompt)

    def test_full_recommended_size_follows_contours_and_pulls(self):
        """At the correct size, a fuller body carries the fabric: it follows the stomach and pulls at its widest point."""
        prompt = build_tryon_prompt(body_profile=_profile("full", 1.1), size_offset=_offset(*PERFECT))
        self.assertIn("the garment follows their real contours", prompt)
        self.assertIn("drapes over its widest point", prompt)
        self.assertIn("Do not smooth the stomach away", prompt)

    def test_shape_block_without_size_uses_recommended_drape(self):
        """With a measured shape but no size selection, the drape for the correct size is still given."""
        prompt = build_tryon_prompt(body_profile=_profile("lean", 0.9))
        self.assertIn("BODY-DEPENDENT DRAPE", prompt)
        self.assertIn("visible slack at the waist and sides", prompt)

    def test_same_size_reads_differently_by_body(self):
        """The same one-size-down garment is described differently on a lean and a very full body."""
        lean = build_tryon_prompt(body_profile=_profile("lean", 0.9), size_offset=_offset(*SNUG))
        full = build_tryon_prompt(body_profile=_profile("very_full", 1.3), size_offset=_offset(*SNUG))
        self.assertIn("over the flat stomach it lies flat without pulling", lean)
        self.assertIn("Strong strain at the widest point of the stomach, far more than on a lean body", full)
        self.assertNotIn("Strong strain at the widest point of the stomach", lean)
        # Both keep the size framing and the frozen body
        for prompt in (lean, full):
            self.assertIn("SNUG, ONE SIZE SMALLER THAN THE VISITOR'S SIZE M", prompt)
            self.assertIn("FROZEN PHYSICAL BODY", prompt)

    def test_strain_shows_where_this_body_is_widest(self):
        """Tight sizes put the tension at the stomach for a fuller body and at the frame for a lean one."""
        full = build_tryon_prompt(body_profile=_profile("full", 1.1), size_offset=_offset(*TIGHT, limiting="waist"))
        lean = build_tryon_prompt(body_profile=_profile("lean", 0.9), size_offset=_offset(*TIGHT, limiting="shoulder"))
        self.assertIn("WHERE THE STRAIN SHOWS: this visitor's body is widest at the stomach", full)
        self.assertIn("WHERE THE STRAIN SHOWS: this visitor's body is widest at the shoulders and upper back", lean)
        self.assertNotIn("widest at the stomach", lean)

    def test_larger_size_shows_more_slack_on_lean_body(self):
        lean = build_tryon_prompt(body_profile=_profile("lean", 0.9), size_offset=_offset(*RELAXED))
        full = build_tryon_prompt(body_profile=_profile("full", 1.1), size_offset=_offset(*RELAXED))
        self.assertIn("obvious empty space at the waist and sides", lean)
        self.assertIn("looks less baggy on this body than it would on a lean one", full)

    def test_slack_scales_with_offset(self):
        texts = [build_tryon_prompt(size_offset=_offset(*f)) for f in (RELAXED, LOOSE, OVERSIZED)]
        self.assertIn("drop about 2–3 cm", texts[0])
        self.assertIn("drop 5–7 cm", texts[1])
        self.assertIn("hang 8–12 cm down the upper arm", texts[2])
        self.assertIn("RELAXED FIT, ONE SIZE LARGER", texts[0])
        self.assertIn("CLEARLY LOOSE, TWO SIZES LARGER", texts[1])
        self.assertIn("CLEARLY BAGGY, SAME DESIGN, THREE SIZES LARGER", texts[2])

    def test_fabric_physics_follow_garment_structure(self):
        stiff = build_tryon_prompt(size_offset=_offset(*LOOSE), cloth_analysis=_garment("rigid_tailored", "blazer"))
        woven = build_tryon_prompt(size_offset=_offset(*LOOSE), cloth_analysis=_garment("semi_structured"))
        fluid = build_tryon_prompt(size_offset=_offset(*LOOSE), cloth_analysis=_garment("soft_drape", "kurti"))
        self.assertIn("Structured tailoring does not collapse into soft folds", stiff)
        self.assertIn("Crisp woven fabric holds some body", woven)
        self.assertIn("Soft, fluid fabric falls under its own weight", fluid)
        tight = build_tryon_prompt(size_offset=_offset(*TIGHT), cloth_analysis=_garment("semi_structured"))
        self.assertIn("Woven fabric has little stretch", tight)
        self.assertIn("overrides any fit wording in the garment notes", tight)

    def test_wont_fit_is_physically_plausible_and_modest(self):
        prompt = build_tryon_prompt(body_profile=_profile("full", 1.1), size_offset=_offset(*WONT_FIT, limiting="waist"))
        self.assertIn("WON'T FIT — TOO SMALL FOR THIS BODY (the visitor's size is M)", prompt)
        self.assertIn("The garment cannot close", prompt)
        self.assertIn("never bare skin", prompt)
        self.assertIn("No floating fabric, no fabric passing through the body", prompt)
        self.assertIn("the body is NOT squeezed smaller", prompt)
        self.assertIn("the fronts cannot close over the belly", prompt)

    def test_rigid_garment_drape_note(self):
        """Structured garments show body volume at the closure; soft garments get no tailoring note."""
        blazer = build_tryon_prompt(body_profile=_profile("full", 1.1), size_offset=_offset(*PERFECT),
                                    cloth_analysis=_garment("rigid_tailored", "blazer"))
        shirt = build_tryon_prompt(body_profile=_profile("full", 1.1), size_offset=_offset(*PERFECT),
                                   cloth_analysis=_garment("soft_drape", "t_shirt"))
        self.assertIn("This is a structured tailored garment: it does not cling", blazer)
        self.assertIn("front button pulling at the waist", blazer)
        self.assertNotIn("structured tailored garment", shirt)

    def test_garment_analysis_injected(self):
        """The garment analysis is part of the prompt (it used to be computed and dropped)."""
        prompt = build_tryon_prompt(cloth_analysis=_garment("rigid_tailored", "blazer"))
        self.assertIn("[GARMENT — Test Garment (blazer)]", prompt)
        self.assertIn("Fabric: Crisp woven cotton", prompt)
        self.assertIn("rigid tailored (holds its own shape); outerwear layered over the inner clothing", prompt)
        self.assertIn("Sleeves: Clean sleeve drape to the wrists", prompt)
        self.assertIn("Try-on guidelines: Keep the placket straight", prompt)

    def test_user_appearance_injected(self):
        """The visitor appearance analysis (including build and midsection) is part of the prompt."""
        appearance = "\n[AUTHENTIC USER IDENTITY & PHOTOGRAPHIC REALISM — STRICTEST REQUIREMENT]\n- Body Build & Midsection (as photographed): stocky; rounded belly."
        prompt = build_tryon_prompt(user_appearance=appearance)
        self.assertIn("AUTHENTIC USER IDENTITY & PHOTOGRAPHIC REALISM", prompt)
        self.assertIn("stocky; rounded belly", prompt)

    def test_no_prompt_references_an_earlier_render(self):
        """Every size anchors to the capture only: no prompt mentions a third image or an earlier render."""
        for fit in (None,) + ALL_FITS:
            prompt = build_tryon_prompt(body_profile=_profile("full", 1.1), cloth_analysis=_garment(),
                                        size_offset=_offset(*fit) if fit else None, selected_size="L")
            self.assertNotIn("Image 3", prompt)
            self.assertNotIn("earlier AI try-on render", prompt)

    def test_strict_retry_protects_midsection(self):
        """The strict retry (sent after a silhouette or midsection failure) names the midsection explicitly."""
        prompt = build_tryon_prompt(retry_strict=True)
        self.assertIn("STRICT SILHOUETTE ALIGNMENT", prompt)
        self.assertIn("do not slim the waist or flatten the stomach", prompt)

    def test_strict_retry_does_not_undo_the_size(self):
        """Locking the body on a retry must not pull a larger size back onto the body."""
        loose = build_tryon_prompt(retry_strict=True, size_offset=_offset(*LOOSE))
        self.assertIn("This locks the BODY only", loose)
        self.assertNotIn("Match the exact outer silhouette boundary", loose)

    def test_fit_escalation_injected(self):
        from app.services.fit_drape import fit_escalation_text
        esc = fit_escalation_text("loose", "L", "looked nearly identical to")
        prompt = build_tryon_prompt(size_offset=_offset(*LOOSE), fit_escalation=esc)
        self.assertIn("SIZE DIFFERENCE NOT VISIBLE — REJECTED", prompt)
        self.assertIn("looked nearly identical to size L", prompt)
        self.assertIn("LOOSER and LARGER", prompt)

    def test_cross_fit_landmark_drift_detection(self):
        """Verify cross-fit landmark regression check catches skeletal drift and passes consistent anatomy."""
        from app.services.silhouette_verifier import silhouette_verifier
        from app.services.body_profile import FrameLandmarks

        baseline = {
            "height_ratio": 0.80,
            "torso_length": 0.32,
            "shoulder_width": 0.28,
            "hip_width": 0.24
        }

        # Consistent anatomy (1% shift) -> Must pass
        consistent_lm = FrameLandmarks(
            height_ratio=0.805,
            shoulder_width=0.282,
            torso_length=0.321,
            hip_width=0.241,
            leg_length=0.45,
            yaw_angle=0.0,
            confidence=0.9,
            keypoints={}
        )
        passed, score, details = silhouette_verifier.verify_cross_fit_landmarks(consistent_lm, baseline, fit_state="fitted")
        self.assertTrue(passed, f"Consistent anatomy should pass. Details: {details}")
        self.assertGreaterEqual(score, 0.90)

        # Distorted height (>8% drift) -> Must fail
        drifted_height_lm = FrameLandmarks(
            height_ratio=0.70,  # 12.5% drift
            shoulder_width=0.28,
            torso_length=0.32,
            hip_width=0.24,
            leg_length=0.40,
            yaw_angle=0.0,
            confidence=0.9,
            keypoints={}
        )
        passed_h, _, details_h = silhouette_verifier.verify_cross_fit_landmarks(drifted_height_lm, baseline, fit_state="relaxed")
        self.assertFalse(passed_h, f"Height drift >8% must fail cross-fit regression. Details: {details_h}")

        # Distorted torso length (>8% drift) -> Must fail
        drifted_torso_lm = FrameLandmarks(
            height_ratio=0.80,
            shoulder_width=0.28,
            torso_length=0.37,  # 15.6% drift
            hip_width=0.24,
            leg_length=0.45,
            yaw_angle=0.0,
            confidence=0.9,
            keypoints={}
        )
        passed_t, _, details_t = silhouette_verifier.verify_cross_fit_landmarks(drifted_torso_lm, baseline, fit_state="fitted")
        self.assertFalse(passed_t, f"Torso drift >8% must fail cross-fit regression. Details: {details_t}")


class TestSizePromptConditioning(unittest.TestCase):
    """Tests for size-specific prompt conditioning injected via size_offset."""

    def _make_size_offset(self, steps, lo_mm, hi_mm, quality, dnf=False):
        return _offset(steps, lo_mm, hi_mm, quality, dnf)

    def test_no_size_offset_no_injection(self):
        """When size_offset is None, no size-specific conditioning should appear."""
        prompt = build_tryon_prompt(fit_state="standard", size_offset=None)
        self.assertNotIn("SIZE-SPECIFIC", prompt)
        self.assertNotIn("RECOMMENDED SIZE", prompt.split("STANDARD")[0] if "STANDARD" in prompt else "")

    def test_perfect_size_injection(self):
        """Recommended size (steps=0) injects the correct-fit section."""
        prompt = build_tryon_prompt(fit_state="standard", size_offset=_offset(*PERFECT), selected_size="M")
        self.assertIn("SIZE M: CORRECT FIT", prompt)
        self.assertIn("15mm to 25mm", prompt)
        self.assertIn("neither tight nor baggy", prompt)

    def test_snug_size_injection(self):
        prompt = build_tryon_prompt(fit_state="standard", size_offset=_offset(*SNUG), selected_size="S")
        self.assertIn("SIZE S: SNUG, ONE SIZE SMALLER", prompt)
        self.assertIn("0mm to 10mm", prompt)
        self.assertIn("Light pulling across the widest part of the torso", prompt)

    def test_tight_size_injection(self):
        prompt = build_tryon_prompt(fit_state="standard", size_offset=_offset(*TIGHT), selected_size="XS")
        self.assertIn("SIZE XS: TIGHT WITH VISIBLE STRAIN, TWO SIZES SMALLER", prompt)
        self.assertIn("X-shaped folds", prompt)
        self.assertIn("The hem rides up 3–5 cm", prompt)

    def test_relaxed_size_injection(self):
        prompt = build_tryon_prompt(fit_state="standard", size_offset=_offset(*RELAXED), selected_size="L")
        self.assertIn("SIZE L: RELAXED FIT, ONE SIZE LARGER", prompt)
        self.assertIn("30mm to 45mm", prompt)

    def test_loose_and_oversized_injection(self):
        loose = build_tryon_prompt(size_offset=_offset(*LOOSE), selected_size="XL")
        oversized = build_tryon_prompt(size_offset=_offset(*OVERSIZED), selected_size="XXL")
        self.assertIn("SIZE XL: CLEARLY LOOSE, TWO SIZES LARGER", loose)
        self.assertIn("SIZE XXL: CLEARLY BAGGY, SAME DESIGN, THREE SIZES LARGER", oversized)
        self.assertNotEqual(loose.split("[SIZE FIT")[1], oversized.split("[SIZE FIT")[1],
                            "+2 and +3 must get different fit text (they used to share one)")

    def test_body_freeze_present_with_all_sizes(self):
        """Body freeze directive must be present regardless of size selection."""
        for fit in ALL_FITS:
            prompt = build_tryon_prompt(fit_state="standard", size_offset=_offset(*fit))
            self.assertIn("FROZEN PHYSICAL BODY", prompt, f"Body freeze missing for quality={fit[3]}")

    def test_body_drape_for_every_size_and_shape(self):
        """Every measured shape gets a drape line for every fit category."""
        for shape in ("lean", "average", "full", "very_full"):
            for fit in ALL_FITS:
                prompt = build_tryon_prompt(body_profile=_profile(shape, 1.0), size_offset=_offset(*fit))
                self.assertIn("BODY-DEPENDENT DRAPE", prompt, f"Drape missing for {shape}/{fit[3]}")

    def test_size_offset_combines_with_body_profile(self):
        """Size offset should appear alongside the measured body shape and its body-dependent drape."""
        prompt = build_tryon_prompt(fit_state="standard", body_profile=_profile("lean", 0.9), size_offset=_offset(*LOOSE))
        self.assertIn("VISITOR BODY SHAPE", prompt)
        self.assertIn("CLEARLY LOOSE", prompt)
        self.assertIn("On this lean body the garment hangs away from the torso on all sides", prompt)


class TestPersonalAppearancePreservation(unittest.TestCase):
    """Every generation carries the personal-appearance rules: only the garment changes."""

    def test_rules_in_every_prompt(self):
        prompt = build_tryon_prompt(size_offset=_offset(*PERFECT), cloth_analysis=_garment(), body_profile=_profile("average"))
        for rule in ("### 1. Face and head (locked)", "its tilt and rotation, and the neck alignment",
                     "facial hair and glasses", "### 2. Body (locked)", "### 3. Garment",
                     "only where it naturally shows through the opening of an open garment",
                     "### 4. Size and fit", "a Medium may be tight on a larger person",
                     "every bindi (regardless of gender), tattoo, piece of jewelry, watch, scar",
                     "must never completely hide a hand or foot", "dupattas, pallus", "Fixed 3:4 portrait",
                     "top of the head 5% below the top edge, soles 4% above the bottom edge",
                     "Seamless clean white studio background", "**HIGHEST PRIORITY:**"):
            self.assertEqual(prompt.count(rule), 1, rule)

    def test_framing_numbers_match_the_framing_applied_afterwards(self):
        self.assertEqual((image_utils.FRAME_TOP, image_utils.FRAME_BOTTOM), (0.05, 0.04))

    def test_no_fixed_meaning_for_a_size_letter(self):
        self.assertNotIn("Medium: regular fit", PROMPT_TEXT)

    def test_inner_clothing_rule_never_keeps_own_sleeves_past_the_garment(self):
        """The old rule kept the visitor's own top 'including sleeves and visible portions', which told the model to keep
        a long sleeve showing below a shorter one. Inner clothing is kept only through a layered garment's opening."""
        self.assertNotIn("sleeves, and visible portions", PROMPT_TEXT)
        self.assertIn("never shows past the selected garment's own sleeves or hem", PROMPT_TEXT)
        self.assertNotIn("visible existing garments must remain unchanged", PROMPT_TEXT)

    def test_outfit_set_bottom_is_part_of_the_garment(self):
        """The priority rule must not stop a kurta / kurti / sherwani set from replacing the visitor's own bottoms."""
        self.assertIn("its matching bottom replaces the person's own lower garment", PROMPT_TEXT)


class TestSherwaniConditioning(unittest.TestCase):
    """The sherwani jacket follows the size fit; its draped dhoti is the same at every size."""

    def test_dhoti_drape_is_identical_at_every_size(self):
        sherwani = _garment("rigid_tailored", "sherwani")
        lines = set()
        for fit in ALL_FITS:
            prompt = build_tryon_prompt(size_offset=_offset(*fit), cloth_analysis=sherwani, body_profile=_profile("average"))
            dhoti = [l for l in prompt.splitlines() if l.startswith("- PARTS THAT DO NOT CHANGE WITH SIZE:")]
            self.assertEqual(len(dhoti), 1)
            lines.add(dhoti[0])
        self.assertEqual(len(lines), 1)
        self.assertIn("gathered at the ankles", lines.pop())

    def test_jacket_still_follows_the_size(self):
        sherwani = _garment("rigid_tailored", "sherwani")
        tight = build_tryon_prompt(size_offset=_offset(*TIGHT), cloth_analysis=sherwani, selected_size="S")
        loose = build_tryon_prompt(size_offset=_offset(*LOOSE), cloth_analysis=sherwani, selected_size="XL")
        self.assertIn("Structured tailoring cannot stretch", tight)
        self.assertIn("Structured tailoring does not collapse into soft folds", loose)

    def test_other_garments_get_no_dhoti_line(self):
        prompt = build_tryon_prompt(size_offset=_offset(*PERFECT), cloth_analysis=_garment("semi_structured", "kurta"))
        self.assertNotIn("PARTS THAT DO NOT CHANGE WITH SIZE", prompt)

    def test_catalog_recognises_the_sherwani_asset(self):
        import os
        from PIL import Image
        from app.services.gemini_client import analyze_cloth
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "assets", "dress", "sherwani1.png")
        if not os.path.exists(path):
            self.skipTest("frontend asset not present in this checkout")
        analysis = analyze_cloth(Image.open(path))  # catalog hit: no API call
        self.assertEqual((analysis.category, analysis.structure_level), ("sherwani", "rigid_tailored"))


class TestKurtaDhotiConditioning(unittest.TestCase):
    """The kurta follows the size fit; the wrapped dhoti is the same at every size."""

    def test_dhoti_drape_is_identical_at_every_size(self):
        kurta_dhoti = _garment("semi_structured", "kurta_dhoti")
        lines = set()
        for fit in ALL_FITS:
            prompt = build_tryon_prompt(size_offset=_offset(*fit), cloth_analysis=kurta_dhoti, body_profile=_profile("full"))
            dhoti = [l for l in prompt.splitlines() if l.startswith("- PARTS THAT DO NOT CHANGE WITH SIZE:")]
            self.assertEqual(len(dhoti), 1)
            lines.add(dhoti[0])
            self.assertIn("VISITOR BODY SHAPE", prompt)
        self.assertEqual(len(lines), 1)
        self.assertIn("front-pleated", lines.pop())

    def test_catalog_recognises_the_kurta_dhoti_asset(self):
        import os
        from PIL import Image
        from app.services.gemini_client import analyze_cloth
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "assets", "dress", "kurta_dhoti1.png")
        if not os.path.exists(path):
            self.skipTest("frontend asset not present in this checkout")
        analysis = analyze_cloth(Image.open(path))  # catalog hit: no API call
        self.assertEqual((analysis.category, analysis.structure_level), ("kurta_dhoti", "semi_structured"))


class TestSalwarSuitConditioning(unittest.TestCase):
    """The kameez follows the size fit; the salwar and dupatta are the same at every size."""

    def test_salwar_and_dupatta_are_identical_at_every_size(self):
        suit = _garment("soft_drape", "salwar_suit")
        lines = set()
        for fit in ALL_FITS:
            prompt = build_tryon_prompt(size_offset=_offset(*fit), cloth_analysis=suit, body_profile=_profile("full"))
            fixed = [l for l in prompt.splitlines() if l.startswith("- PARTS THAT DO NOT CHANGE WITH SIZE:")]
            self.assertEqual(len(fixed), 1)
            lines.add(fixed[0])
            self.assertIn("VISITOR BODY SHAPE", prompt)
        self.assertEqual(len(lines), 1)
        line = lines.pop()
        self.assertIn("salwar", line)
        self.assertIn("dupatta", line)
        self.assertIn("Only the kameez follows the size fit", line)

    def test_catalog_recognises_the_salwar_suit_asset(self):
        import os
        from PIL import Image
        from app.services.gemini_client import analyze_cloth
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "assets", "dress", "salwar_suit1.png")
        if not os.path.exists(path):
            self.skipTest("frontend asset not present in this checkout")
        analysis = analyze_cloth(Image.open(path))  # catalog hit: no API call
        self.assertEqual((analysis.category, analysis.structure_level), ("salwar_suit", "soft_drape"))

    def test_salwar_suit_dupatta_keeps_both_hands_visible(self):
        """The catalog photo drapes the dupatta over the left cuff; copying that buried the visitor's hand."""
        from app.services.gemini_client import _PRE_REGISTERED_CATALOG
        guidelines = _PRE_REGISTERED_CATALOG["d1893e3bbb81c513"].tryon_guidelines
        self.assertIn("do NOT copy that part of the drape", guidelines)
        self.assertIn("whole left sleeve, cuff and hand stay outside it", guidelines)
        self.assertIn("The wearer's right arm and hand (viewer's left) are equally clear", guidelines)

    def test_no_prompt_tells_the_model_to_copy_a_drape_over_the_hands(self):
        prompt = build_tryon_prompt(size_offset=_offset(*RELAXED), cloth_analysis=_garment(), body_profile=_profile("average"))
        self.assertNotIn("draped exactly as in Image 2", prompt)
        self.assertIn("hangs clear of both hands and forearms", prompt)
        self.assertIn("never covers a hand or forearm", prompt)


class TestKurtaPajamaConditioning(unittest.TestCase):
    """Unlike the dhoti sets, the straight-cut pants are sized with the kurta: they change with the selected size."""

    def test_pants_follow_the_size(self):
        outfit = _garment("semi_structured", "kurta_pajama")
        lines = {}
        for fit in ALL_FITS:
            prompt = build_tryon_prompt(size_offset=_offset(*fit), cloth_analysis=outfit, body_profile=_profile("average"))
            pants = [l for l in prompt.splitlines() if l.startswith("- PANTS AT THIS SIZE:")]
            self.assertEqual(len(pants), 1)
            self.assertNotIn("PARTS THAT DO NOT CHANGE WITH SIZE", prompt)
            self.assertIn("VISITOR BODY SHAPE", prompt)
            lines[fit[3]] = pants[0]
        self.assertEqual(len(set(lines.values())), len(ALL_FITS))   # a different pants fit at every size
        self.assertIn("ankle bone", lines["perfect"])
        self.assertIn("deep break on the top of the feet", lines["oversized"])
        self.assertIn("above the ankle", lines["tight"])

    def test_other_garments_get_no_pants_line(self):
        for category in ("kurta", "kurta_dhoti", "shirt"):
            prompt = build_tryon_prompt(size_offset=_offset(*LOOSE), cloth_analysis=_garment("semi_structured", category))
            self.assertNotIn("PANTS AT THIS SIZE", prompt)

    def test_catalog_recognises_the_kurta_pajama_asset(self):
        import os
        from PIL import Image
        from app.services.gemini_client import analyze_cloth
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "assets", "dress", "kurta_pajama1.png")
        if not os.path.exists(path):
            self.skipTest("frontend asset not present in this checkout")
        analysis = analyze_cloth(Image.open(path))  # catalog hit: no API call
        self.assertEqual((analysis.category, analysis.structure_level), ("kurta_pajama", "semi_structured"))


class _PerSizeLines(unittest.TestCase):
    def _lines(self, category, structure, prefix):
        """The one prompt line starting with `prefix` at every fit category, for a fuller body."""
        lines = {}
        for fit in ALL_FITS:
            prompt = build_tryon_prompt(size_offset=_offset(*fit), cloth_analysis=_garment(structure, category),
                                        body_profile=_profile("full"))
            self.assertIn("VISITOR BODY SHAPE", prompt)
            found = [l for l in prompt.splitlines() if l.startswith(prefix)]
            self.assertEqual(len(found), 1, (category, fit[3], prefix))
            lines[fit[3]] = found[0]
        return lines


class TestLehengaAndRoyalSherwaniConditioning(_PerSizeLines):
    """Fitted parts follow the size; the flared skirt, dupatta, stole and turban are the same at every size."""

    def test_lehenga_skirt_and_dupatta_fixed_blouse_sized(self):
        fixed = self._lines("lehenga", "semi_structured", "- PARTS THAT DO NOT CHANGE WITH SIZE:")
        self.assertEqual(len(set(fixed.values())), 1)
        line = fixed["perfect"]
        self.assertIn("skirt", line)
        self.assertIn("dupatta", line)
        self.assertIn("Only the fitted blouse follows the size fit", line)
        prompt = build_tryon_prompt(size_offset=_offset(*LOOSE), cloth_analysis=_garment("semi_structured", "lehenga"))
        self.assertNotIn("PANTS AT THIS SIZE", prompt)

    def test_royal_sherwani_accessories_fixed_churidar_sized(self):
        fixed = self._lines("royal_sherwani", "rigid_tailored", "- PARTS THAT DO NOT CHANGE WITH SIZE:")
        self.assertEqual(len(set(fixed.values())), 1)
        self.assertIn("stole", fixed["perfect"])
        self.assertIn("turban", fixed["perfect"])
        churidar = self._lines("royal_sherwani", "rigid_tailored", "- PANTS AT THIS SIZE:")
        self.assertEqual(len(set(churidar.values())), len(ALL_FITS))   # a different churidar fit at every size
        self.assertIn("churidar", churidar["perfect"])
        self.assertIn("gathers", churidar["loose"])

    def test_catalog_recognises_both_assets(self):
        import os
        from PIL import Image
        from app.services.gemini_client import analyze_cloth
        base = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "assets", "dress")
        for name, expected in (("lehenga1.png", ("lehenga", "semi_structured")),
                               ("royal_sherwani1.png", ("royal_sherwani", "rigid_tailored"))):
            path = os.path.join(base, name)
            if not os.path.exists(path):
                self.skipTest("frontend asset not present in this checkout")
            analysis = analyze_cloth(Image.open(path))  # catalog hit: no API call
            self.assertEqual((analysis.category, analysis.structure_level), expected)


class TestSareeAndMunduConditioning(_PerSizeLines):
    """The blouse / shirt follow the size; the saree drape, mundu, shawl and necklace are the same at every size."""

    def test_catalog_recognises_both_assets(self):
        import os
        from PIL import Image
        from app.services.gemini_client import analyze_cloth
        base = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "assets", "dress")
        for name, expected in (("kerala_saree1.png", ("saree", "semi_structured")),
                               ("mundu_shirt1.png", ("mundu_shirt", "semi_structured"))):
            path = os.path.join(base, name)
            if not os.path.exists(path):
                self.skipTest("frontend asset not present in this checkout")
            analysis = analyze_cloth(Image.open(path))  # catalog hit: no API call
            self.assertEqual((analysis.category, analysis.structure_level), expected)

    def test_saree_drape_fixed_blouse_sized(self):
        fixed = self._lines("saree", "semi_structured", "- PARTS THAT DO NOT CHANGE WITH SIZE:")
        self.assertEqual(len(set(fixed.values())), 1)
        self.assertIn("pallu", fixed["perfect"])
        self.assertIn("Only the fitted green blouse follows the size fit", fixed["perfect"])

    def test_mundu_and_shawl_fixed_shirt_sized(self):
        fixed = self._lines("mundu_shirt", "semi_structured", "- PARTS THAT DO NOT CHANGE WITH SIZE:")
        self.assertEqual(len(set(fixed.values())), 1)
        self.assertIn("mundu", fixed["perfect"])
        self.assertIn("shawl", fixed["perfect"])
        self.assertIn("Only the shirt follows the size fit", fixed["perfect"])
        prompt = build_tryon_prompt(size_offset=_offset(*LOOSE), cloth_analysis=_garment("semi_structured", "mundu_shirt"))
        self.assertNotIn("PANTS AT THIS SIZE", prompt)
        self.assertIn("CLEARLY LOOSE", prompt)   # the shirt still gets the full larger-size fit


if __name__ == "__main__":
    unittest.main()
