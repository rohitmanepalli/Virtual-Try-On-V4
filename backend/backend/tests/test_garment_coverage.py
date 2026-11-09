"""
Garment coverage: where the selected garment ends, the visitor's own clothing must not continue, and where it reaches
further than the visitor's own sleeves, no bare skin may show. There is no inpaint mask (the model redraws the whole
frame), so this is guarded by the prompt's coverage section and by a pixel check on every render.
"""
import asyncio
import base64
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.models import ClothAnalysis, SilhouetteVerificationResult, TryOnRequest
from app.routers import tryon
from app.services import garment_coverage as gc
from app.services.gemini_client import (
    _PRE_REGISTERED_CATALOG, _USER_APPEARANCE_CACHE, _USER_APPEARANCE_DETAILS, analyze_user_appearance,
    build_tryon_prompt, get_appearance_details,
)

W, H = 400, 600
SKIN = (190, 140, 110)
OLD_RED = (150, 30, 40)  # the visitor's own long sleeve
NAVY = (30, 50, 90)  # a new full sleeve
WHITE = (240, 240, 240)  # a new three-quarter sleeve
WATCH = (20, 20, 20)

KEYPOINTS = {
    "left_shoulder": (0.62, 0.25, 0.0), "left_elbow": (0.66, 0.45, 0.0), "left_wrist": (0.68, 0.62, 0.0),
    "left_index": (0.69, 0.70, 0.0),
    "right_shoulder": (0.38, 0.25, 0.0), "right_elbow": (0.34, 0.45, 0.0), "right_wrist": (0.32, 0.62, 0.0),
    "right_index": (0.31, 0.70, 0.0),
}


def _lab(rgb):
    return gc._lab(np.array([[rgb]], dtype=np.uint8))[0].tolist()


def _lm(drop=()):
    kp = {k: v for k, v in KEYPOINTS.items() if k not in drop}
    return SimpleNamespace(keypoints=kp, visibility={k: 1.0 for k in kp}, segmentation_mask=None)


def _px(name):
    x, y, _ = KEYPOINTS[name]
    return np.array([x * W, y * H])


def _paint(img, start, end, t0, t1, color, thickness=26):
    a, b = _px(start), _px(end)
    p0, p1 = a + (b - a) * t0, a + (b - a) * t1
    cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), color, thickness)


def _render(forearm=SKIN, upper=WHITE, sleeve_to=None, sleeve=None, watch=False):
    """Both arms: upper arm, forearm, hand (skin). sleeve_to/sleeve: a sleeve covering the forearm up to that fraction."""
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    for side in ("left", "right"):
        _paint(img, f"{side}_shoulder", f"{side}_elbow", 0.0, 1.0, upper)
        _paint(img, f"{side}_elbow", f"{side}_wrist", 0.0, 1.0, forearm)
        if sleeve_to is not None:
            _paint(img, f"{side}_elbow", f"{side}_wrist", 0.0, sleeve_to, sleeve)
        if watch:
            _paint(img, f"{side}_elbow", f"{side}_wrist", 0.92, 1.0, WATCH)
        _paint(img, f"{side}_wrist", f"{side}_index", 0.0, 1.0, SKIN, thickness=34)
    return img


def _capture(own="full", palette=(OLD_RED,)):
    return gc.CaptureCoverage(own_sleeves=own, skin_lab=_lab(SKIN), skin_hex="#BE8C6E", hand_lab=_lab(SKIN),
                              sleeve_colors={s: [_lab(c) for c in palette] for s in ("left", "right")} if palette else {})


def _check(img, capture, garment_sleeve, lm=None):
    return gc.check_arms(img, lm or _lm(), capture, garment_sleeve, np.array(_lab(SKIN)), (_lab(SKIN), _lab(SKIN)))


class RemnantCheckTest(unittest.TestCase):
    """Visitor in long sleeves, garment with shorter sleeves: the old sleeve must be gone below the new one."""

    def test_old_sleeve_on_the_forearm_fails(self):
        res = _check(_render(forearm=OLD_RED), _capture(), "short")
        self.assertIs(res["passed"], False)
        self.assertEqual(res["problem"], "remnant")
        self.assertEqual(res["sides"], "left and right")

    def test_bare_forearms_pass(self):
        res = _check(_render(), _capture(), "short")
        self.assertIs(res["passed"], True)
        self.assertLess(res["arms"]["left"]["remnant_share"], 0.05)

    def test_watch_at_the_wrist_is_not_a_remnant(self):
        """The judged stretch stops short of the wrist, so a dark watch never reads as a dark old sleeve."""
        res = _check(_render(watch=True), _capture(palette=(WATCH, OLD_RED)), "short")
        self.assertIs(res["passed"], True)

    def test_three_quarter_sleeve_judges_only_below_it(self):
        clean = _check(_render(sleeve_to=0.55, sleeve=WHITE), _capture(), "three_quarter")
        self.assertIs(clean["passed"], True)
        leftover = _check(_render(forearm=OLD_RED, sleeve_to=0.55, sleeve=WHITE), _capture(), "three_quarter")
        self.assertIs(leftover["passed"], False)

    def test_one_arm_with_a_remnant_fails_the_render(self):
        img = _render()
        _paint(img, "left_elbow", "left_wrist", 0.0, 1.0, OLD_RED)
        res = _check(img, _capture(), "short")
        self.assertIs(res["passed"], False)
        self.assertEqual(res["sides"], "left")


class GapCheckTest(unittest.TestCase):
    """Visitor in short sleeves, garment with sleeves to the wrist: no bare forearm may show."""

    def test_bare_forearm_under_a_full_sleeve_fails(self):
        res = _check(_render(forearm=SKIN, upper=NAVY), _capture(own="short", palette=()), "full")
        self.assertIs(res["passed"], False)
        self.assertEqual(res["problem"], "gap")

    def test_covered_forearm_passes(self):
        res = _check(_render(forearm=NAVY, upper=NAVY), _capture(own="short", palette=()), "full")
        self.assertIs(res["passed"], True)

    def test_skin_coloured_new_sleeve_is_not_judged(self):
        res = _check(_render(forearm=SKIN, upper=(195, 142, 112)), _capture(own="short", palette=()), "full")
        self.assertIsNone(res["passed"])


class NotCheckedTest(unittest.TestCase):
    """Anything unmeasurable is 'not checked' and never fails a render."""

    def test_old_sleeve_colour_close_to_skin(self):
        res = _check(_render(forearm=(196, 146, 114)), _capture(palette=((196, 146, 114),)), "short")
        self.assertIsNone(res["passed"])

    def test_missing_wrist_landmarks(self):
        res = _check(_render(forearm=OLD_RED), _capture(), "short", lm=_lm(drop=("left_wrist", "right_wrist")))
        self.assertIsNone(res["passed"])

    def test_unknown_own_sleeves(self):
        res = _check(_render(forearm=OLD_RED), _capture(own=None), "short")
        self.assertIsNone(res["passed"])
        self.assertEqual(res["reason"], "visitor's own sleeve length unknown")

    def test_unknown_garment_sleeves(self):
        self.assertIsNone(_check(_render(forearm=OLD_RED), _capture(), None)["passed"])

    def test_same_sleeve_length(self):
        self.assertIsNone(_check(_render(forearm=OLD_RED), _capture(own="short"), "short")["passed"])

    def test_check_render_compares_hand_to_hand(self):
        """Face and arms are lit differently in both images; the hands are lit like the forearms."""
        res = gc.check_render(_capture(), Image.fromarray(_render(forearm=OLD_RED)), _lm(), "short")
        self.assertEqual(res["skin_reference"], "hands")
        self.assertIs(res["passed"], False)

    def test_capture_hands_out_of_frame_still_judged_on_the_render_hands(self):
        """Captures cropped above the knees often cut off the hands; the render's own hands are still the reference."""
        cap = _capture()
        cap.hand_lab, cap.skin_lab = None, None
        res = gc.check_render(cap, Image.fromarray(_render(forearm=OLD_RED)), _lm(), "short")
        self.assertEqual(res["skin_reference"], "hands")
        self.assertIs(res["passed"], False)
        clean = gc.check_render(cap, Image.fromarray(_render()), _lm(), "short")
        self.assertIs(clean["passed"], True)


class CoverageSectionTest(unittest.TestCase):
    DHOTI = _PRE_REGISTERED_CATALOG["65a832f07fadc2af"]  # three-quarter sleeves, mid-thigh kurta

    def test_long_sleeved_visitor_three_quarter_garment(self):
        cap = gc.CaptureCoverage(own_sleeves="full", skin_hex="#AF8D85",
                                 wrist_accessories="black wristwatch on the wrist at image right")
        text = gc.coverage_section(self.DHOTI, cap)
        self.assertIn("sleeves end at mid-forearm", text)
        self.assertIn("long sleeves to the wrist, longer than the selected garment's", text)
        self.assertIn("remove them completely", text)
        self.assertIn("about #AF8D85", text)
        self.assertIn("WRIST ACCESSORIES IN IMAGE 1: black wristwatch on the wrist at image right.", text)
        self.assertIn("ends at mid-thigh", text)
        self.assertIn("necklace or chain", text)

    def test_short_sleeved_visitor_full_sleeve_garment(self):
        shirt = _PRE_REGISTERED_CATALOG["da8dff6edab84939"]
        text = gc.coverage_section(shirt, gc.CaptureCoverage(own_sleeves="short"))
        self.assertIn("run to the wrist", text)
        self.assertIn("short sleeves, shorter than the selected garment's", text)
        self.assertIn("cover that bare skin completely", text)
        self.assertNotIn("remove them completely", text)

    def test_unknown_visitor_sleeves_get_the_generic_rule(self):
        text = gc.coverage_section(self.DHOTI, None)
        self.assertIn("If the person wears longer sleeves in Image 1, remove them completely", text)
        self.assertIn("WRIST ACCESSORIES: any watch", text)

    def test_cropped_top(self):
        lehenga = _PRE_REGISTERED_CATALOG["a8f933cee637ac41"]
        self.assertIn("bare midriff skin", gc.coverage_section(lehenga, gc.CaptureCoverage(own_sleeves="full")))

    def test_lower_body_rule_always_present(self):
        cloth = ClothAnalysis(category="shirt", garment_name="X", fabric_type="cotton", sleeve_behavior="n",
                              tryon_guidelines="n")
        section = gc.coverage_section(cloth, None)
        self.assertIn("LOWER BODY", section)
        self.assertIn("never bare", section)

    def test_lower_body_by_capture_bottom(self):
        shirt = ClothAnalysis(category="shirt", garment_name="X", fabric_type="cotton", sleeve_behavior="n",
                              tryon_guidelines="n", sleeve_length="full")
        keep = gc.coverage_section(shirt, gc.CaptureCoverage(own_bottom="separate"))
        self.assertIn("KEEP THE PERSON'S OWN", keep)
        frock = gc.coverage_section(shirt, gc.CaptureCoverage(own_bottom="one_piece"))
        self.assertIn("NEW PLAIN BOTTOM", frock)
        self.assertIn("trousers", frock)
        tee = shirt.model_copy(update={"category": "t_shirt"})
        self.assertIn("jeans", gc.coverage_section(tee, gc.CaptureCoverage(own_bottom="one_piece")))
        dress = shirt.model_copy(update={"includes_bottom": True})
        self.assertIn("includes its own lower garment", gc.coverage_section(dress, gc.CaptureCoverage(own_bottom="separate")))

    def test_prompt_carries_the_section_and_the_retry_directive(self):
        cap = gc.CaptureCoverage(own_sleeves="full")
        retry = gc.coverage_escalation_text({"problem": "remnant", "sides": "left"}, "three_quarter")
        prompt = build_tryon_prompt(cloth_analysis=self.DHOTI, capture_coverage=cap, coverage_escalation=retry)
        self.assertEqual(prompt.count("[GARMENT COVERAGE"), 1)
        self.assertLess(prompt.index("[GARMENT —"), prompt.index("[GARMENT COVERAGE"))
        self.assertIn("own longer sleeve below the selected garment's three-quarter sleeve on the left forearm", prompt)

    def test_gap_retry_directive(self):
        self.assertIn("bare skin on the right forearm", gc.coverage_escalation_text({"problem": "gap", "sides": "right"}, "full"))


class GarmentLengthsTest(unittest.TestCase):
    def test_every_catalog_garment_has_its_lengths(self):
        """A garment without lengths silently skips the coverage section and check: add both with the garment."""
        for key, cloth in _PRE_REGISTERED_CATALOG.items():
            self.assertIn(cloth.sleeve_length, gc.SLEEVE_LENGTHS, key)
            self.assertIn(cloth.top_length, gc.TOP_LENGTHS, key)

    def test_analysis_answers_are_normalised(self):
        for raw, want in [("Full", "full"), ("long sleeves", "full"), ("3/4", "three_quarter"), ("three-quarter", "three_quarter"),
                          ("half sleeve", "short"), ("rolled up", "elbow"), ("midriff", "cropped"), ("ankle", "below_knee"),
                          ("cape", None), (None, None)]:
            allowed = gc.TOP_LENGTHS if want in gc.TOP_LENGTHS else gc.SLEEVE_LENGTHS
            self.assertEqual(gc.normalize_length(raw, allowed), want, raw)

    def test_appearance_analysis_records_own_sleeves_and_wrists(self):
        answer = {"gender_presentation": "Male", "own_sleeves": "long sleeves",
                  "wrist_accessories": "black watch on the wrist at image right", "own_bottom": "one-piece"}
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(
            text=json.dumps(answer), usage_metadata=None)))
        img = Image.new("RGB", (60, 80), (123, 45, 67))
        try:
            analyze_user_appearance(img, client=client)
            self.assertEqual(get_appearance_details(img),
                             {"own_sleeves": "full", "wrist_accessories": "black watch on the wrist at image right",
                              "own_bottom": "one_piece"})
        finally:
            _USER_APPEARANCE_CACHE.clear()
            _USER_APPEARANCE_DETAILS.clear()

    def test_no_wrist_accessories_is_none(self):
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(
            text=json.dumps({"own_sleeves": "short", "wrist_accessories": "none"}), usage_metadata=None)))
        img = Image.new("RGB", (60, 80), (12, 45, 67))
        try:
            analyze_user_appearance(img, client=client)
            self.assertIsNone(get_appearance_details(img)["wrist_accessories"])
        finally:
            _USER_APPEARANCE_CACHE.clear()
            _USER_APPEARANCE_DETAILS.clear()


def _jpeg(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _uri(arr):
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(arr)).decode()


STUDIO = np.full((512, 384, 3), 238, dtype=np.uint8)
cv2.rectangle(STUDIO, (134, 77), (250, 486), (90, 60, 40), -1)
SHORT_SLEEVED = ClothAnalysis(category="t_shirt", garment_name="Polo", fabric_type="knit", structure_level="soft_drape",
                              sleeve_behavior="short", tryon_guidelines="n", sleeve_length="short", top_length="hip")
PASSING = SilhouetteVerificationResult(score=0.9, passed=True, threshold=0.6, details={})
REMNANT = {"checked": True, "passed": False, "problem": "remnant", "sides": "left", "arms": {}}
CLEAN = {"checked": True, "passed": True, "problem": None, "sides": None, "arms": {}}
UNMEASURED = {"checked": False, "passed": None, "problem": None, "sides": None, "arms": {}}


class CoverageRetryTest(unittest.TestCase):
    """A render showing the visitor's own sleeve past the garment's gets one retry with the coverage directive."""

    def _run(self, verdicts):
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return _jpeg(STUDIO), 0.97, SHORT_SLEEVED

        req = TryOnRequest(user_image=_uri(STUDIO), dress_image=_uri(STUDIO), cloth_analysis=SHORT_SLEEVED)
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", False), \
             patch.object(tryon, "generate_tryon_image", side_effect=fake_generate), \
             patch.object(tryon, "analyze_user_appearance", return_value=""), \
             patch.object(tryon, "read_capture", return_value=gc.CaptureCoverage(own_sleeves="full")), \
             patch.object(tryon, "check_render", side_effect=list(verdicts)), \
             patch.object(tryon, "_capture_has_face", return_value=True), \
             patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im), \
             patch.object(tryon.silhouette_verifier, "verify", return_value=PASSING), \
             patch.object(tryon.body_profile_service, "extract_single_frame", return_value=None), \
             patch.object(tryon.body_profile_service, "extract_profile", side_effect=RuntimeError("n/a")), \
             patch.object(tryon.body_profile_service, "generate_face_and_hands_mask",
                          return_value=(None, np.zeros((512, 384), np.float32))), \
             patch.object(tryon, "_face_score", return_value=0.99), \
             patch.object(tryon, "apply_watermark"), \
             patch.object(tryon, "generate_qr_data_uri", return_value="qr"):
            return asyncio.run(tryon.virtual_tryon(req)), calls

    def test_remnant_is_regenerated_with_the_coverage_directive(self):
        resp, calls = self._run([REMNANT, CLEAN])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["coverage_escalation"])
        self.assertIn("COVERAGE RETRY", calls[1]["coverage_escalation"])
        self.assertIsInstance(calls[0]["capture_coverage"], gc.CaptureCoverage)
        self.assertTrue(resp.success)
        self.assertIs(resp.coverage_verified, True)

    def test_still_failing_after_the_retry_is_served_flagged(self):
        resp, calls = self._run([REMNANT, REMNANT])
        self.assertEqual(len(calls), 2)  # one coverage retry, then the applicable retries are used up
        self.assertTrue(resp.success)
        self.assertIs(resp.coverage_verified, False)

    def test_unmeasurable_render_is_not_retried(self):
        resp, calls = self._run([UNMEASURED])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(resp.coverage_verified)


if __name__ == "__main__":
    unittest.main()
