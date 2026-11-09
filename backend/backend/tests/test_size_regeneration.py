"""
Size changes are full Gemini regenerations from the capture and garment only (never an earlier render), classified by
offset and body check, verified against the other sizes, and "won't fit" is its own single-attempt state.
"""
import asyncio
import base64
import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.models import BodyProfile, ClothAnalysis, SilhouetteVerificationResult, TryOnRequest
from app.routers import tryon
from app.services.session_manager import session_manager
from app.services.size_chart import ALL_SIZES, SIZE_INDEX, recommend_size

W, H = 384, 512


def studio(color):
    img = np.full((H, W, 3), 238, np.uint8)
    cv2.rectangle(img, (130, 70), (250, 490), color, -1)
    return img


def to_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_bmp(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="BMP")
    return buf.getvalue()


def metrics(torso, outer=None, look=0.0):
    """Garment silhouette stand-in; renders with the same `look` have an identical torso appearance."""
    return {"torso_width": torso, "outer_width": outer if outer is not None else torso + 0.3, "torso_rows": 20,
            "appearance": [look] * 8}


def passing(midsection_preserved=True, score=0.9):
    details = {"midsection_preserved": midsection_preserved}
    if not midsection_preserved:
        details["midsection_change"] = -0.15
    return SilhouetteVerificationResult(score=score, passed=midsection_preserved and score >= 0.6, threshold=0.6,
                                        details=details)


class SizeRegenerationBase(unittest.TestCase):
    structure = "soft_drape"

    def setUp(self):
        self.capture = studio((90, 60, 40))
        self.cloth = ClothAnalysis(category="shirt", garment_name="Shirt", fabric_type="cotton", structure_level=self.structure,
                                   is_outerwear=False, sleeve_behavior="n", tryon_guidelines="n")
        self.profile = BodyProfile(height_ratio=0.8, shoulder_width_ratio=0.26, torso_aspect_ratio=1.1, shoulder_to_hip_ratio=1.1,
                                   chest_proportion=1.25, waist_proportion=0.85, hip_proportion=1.0, body_type="Rectangle / Balanced",
                                   torso_fullness=1.1, torso_shape="full")
        self.recommended = recommend_size(self.profile, garment_category="shirt").recommended_size
        self.session_id = f"size-regen-test-{self._testMethodName}"
        self.calls = []
        self.silhouettes = []   # garment_silhouette result per generation, in order
        self.verdicts = []      # verify() result per generation, in order
        self.addCleanup(session_manager.purge_session, self.session_id)

    def _generate(self, **kw):
        self.calls.append(kw)
        shade = 40 + 30 * len(self.calls)
        return to_bmp(studio((shade, 40, 160))), 0.97, self.cloth

    def _request(self, size):
        req = TryOnRequest(user_image=to_uri(self.capture), dress_image=to_uri(self.capture), cloth_analysis=self.cloth,
                           body_profile=self.profile, session_id=self.session_id, selected_size=size)
        n = lambda seq, default: seq[len(self.calls) - 1] if len(self.calls) - 1 < len(seq) else default  # noqa: E731
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", False), \
             patch.object(tryon, "_capture_has_face", return_value=True), \
             patch.object(tryon, "generate_tryon_image", side_effect=self._generate), \
             patch.object(tryon, "analyze_user_appearance", return_value=""), \
             patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im), \
             patch.object(tryon.silhouette_verifier, "verify", side_effect=lambda **kw: n(self.verdicts, passing())), \
             patch.object(tryon, "garment_silhouette", side_effect=lambda img, lm: n(self.silhouettes, None)), \
             patch.object(tryon.body_profile_service, "extract_single_frame", return_value=None), \
             patch.object(tryon.body_profile_service, "generate_face_and_hands_mask",
                          return_value=(None, np.zeros((H, W), np.float32))), \
             patch.object(tryon, "_face_score", return_value=0.99), \
             patch.object(tryon, "apply_watermark"), \
             patch.object(tryon, "generate_qr_data_uri", return_value="qr"):
            return asyncio.run(tryon.virtual_tryon(req))

    def size(self, steps):
        return ALL_SIZES[SIZE_INDEX[self.recommended] + steps]


class TestSizeRegeneration(SizeRegenerationBase):
    def test_every_size_is_generated_from_capture_and_garment_only(self):
        first = self._request(None)
        self.assertEqual(first.selected_size, self.recommended)
        self.assertEqual(first.fit_label, "Correct fit")
        second = self._request(self.size(1))
        self.assertEqual(len(self.calls), 2, "a size change must be a new Gemini generation")
        call = self.calls[1]
        self.assertEqual(call["selected_size"], self.size(1))
        self.assertEqual(call["size_offset"].fit_quality, "relaxed")
        self.assertEqual(call["body_profile"].torso_shape, "full")
        for c in self.calls:
            self.assertNotIn("reference_render", c, "no earlier render may be a generation input")
        self.assertEqual(second.size_offset, 1)
        self.assertEqual(second.fit_label, "Relaxed fit")
        self.assertEqual(set(second.size_recommendation.size_fits), set(ALL_SIZES))

    def test_later_sizes_get_the_first_render_as_face_lock_source_only(self):
        self._request(None)
        self._request(self.size(1))
        self.assertIsNone(self.calls[0]["identity_face"])
        self.assertIsNotNone(self.calls[1]["identity_face"], "first served render supplies the face for later sizes")

    def test_revisited_size_is_served_from_cache(self):
        self._request(None)
        self._request(self.size(1))
        again = self._request(self.size(1))
        self.assertEqual(len(self.calls), 2, "a size already rendered in this session must not be regenerated")
        self.assertTrue(again.success)
        self.assertEqual(again.fit_label, "Relaxed fit")

    def test_three_sizes_up_is_very_oversized_not_clamped_to_two(self):
        self._request(None)
        self._request(self.size(2))
        self._request(self.size(3))
        self.assertEqual(self.calls[1]["size_offset"].fit_quality, "loose")
        self.assertEqual(self.calls[2]["size_offset"].fit_quality, "oversized")


class TestCrossSizeVerification(SizeRegenerationBase):
    def test_near_identical_size_is_a_failed_generation_and_retried(self):
        # recommended render, then a larger size whose first attempt looks the same (silhouette and torso)
        self.silhouettes = [metrics(1.00), metrics(1.02), metrics(1.08, look=0.5)]
        self._request(None)
        resp = self._request(self.size(1))
        self.assertEqual(len(self.calls), 3, "the near-identical attempt must be regenerated")
        self.assertIsNone(self.calls[1]["fit_escalation"])
        self.assertIn("looked nearly identical to size", self.calls[2]["fit_escalation"])
        self.assertGreater(self.calls[2]["attempt"], self.calls[1]["attempt"])
        self.assertTrue(resp.fit_verified)

    def test_noise_level_width_reading_is_not_retried(self):
        """A smaller size reading slightly wider (within the measured ~6% noise) is not treated as a failure."""
        self.silhouettes = [metrics(1.00), metrics(1.05, look=0.5)]
        self._request(None)
        resp = self._request(self.size(-1))
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(resp.fit_verified)

    def test_grossly_wider_tight_size_is_retried(self):
        self.silhouettes = [metrics(1.00), metrics(1.09, look=0.5), metrics(0.99, look=0.5)]
        self._request(None)
        self._request(self.size(-1))
        self.assertEqual(len(self.calls), 3)
        self.assertIn("TIGHTER and SMALLER", self.calls[2]["fit_escalation"])

    def test_unfixable_size_serves_the_most_distinct_attempt_flagged(self):
        self.silhouettes = [metrics(1.00), metrics(1.00, look=0.05), metrics(1.00, look=0.20)]
        self._request(None)
        resp = self._request(self.size(1))
        self.assertEqual(len(self.calls), 3, "one fit retry only (budget = GENERATION_MAX_ATTEMPTS)")
        self.assertTrue(resp.success)
        self.assertFalse(resp.fit_verified)
        session = session_manager.get_session(self.session_id)
        dress_hash = tryon.get_dress_hash(Image.fromarray(self.capture))[:16]
        self.assertEqual(resp.result_image, session.get_size_render(self.size(1), dress_hash))
        # attempt 3 (look 0.20) is less like the recommended render than attempt 2 (look 0.05)
        self.assertAlmostEqual(session.get_size_render_metrics(dress_hash)[self.size(1)]["appearance"][0], 0.20)

    def test_midsection_check_runs_on_every_size_and_triggers_strict_retry(self):
        self.verdicts = [passing(), passing(midsection_preserved=False), passing()]
        self._request(None)
        self._request(self.size(2))
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(self.calls[2]["retry_strict"])


class TestWontFit(SizeRegenerationBase):
    structure = "semi_structured"  # typical M body: XS in a woven shirt is past the stretch allowance

    def test_wont_fit_renders_once_with_message(self):
        self._request(None)
        resp = self._request(self.size(-2))
        self.assertEqual(len(self.calls), 2, "won't fit is rendered, not short-circuited")
        self.assertEqual(self.calls[1]["size_offset"].fit_quality, "does_not_fit")
        self.assertTrue(resp.wont_fit)
        self.assertFalse(resp.illustration_only)
        self.assertIsNotNone(resp.result_image)
        self.assertEqual(resp.fit_label, "Won't fit")
        self.assertEqual(resp.message, f"This size won't fit you. We recommend {self.recommended}.")
        self.assertIn("stomach", resp.fit_check["strain_zones"])

    def test_wont_fit_is_never_retried_or_sent_to_realign(self):
        """A strained render with a low silhouette score is intentional: no retry, no 'adjust your stance'."""
        self.verdicts = [passing(), passing(score=0.50)]
        self._request(None)
        resp = self._request(self.size(-2))
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(resp.needs_visitor_realignment)
        self.assertTrue(resp.success and resp.wont_fit)
        self.assertIsNotNone(resp.result_image)

    def test_unbelievable_wont_fit_render_becomes_message_and_illustration(self):
        """A render that squeezed the body is not shown; the message + illustration are, and cached (no repeat cost)."""
        self.verdicts = [passing(), passing(midsection_preserved=False)]
        self._request(None)
        resp = self._request(self.size(-2))
        self.assertEqual(len(self.calls), 2, "no retry for won't fit")
        self.assertTrue(resp.success and resp.wont_fit and resp.illustration_only)
        self.assertIsNone(resp.result_image)
        self.assertIn("squeezed", resp.fit_check["render_rejected"])
        again = self._request(self.size(-2))
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(again.illustration_only)

    def test_wont_fit_render_never_becomes_fallback_or_face(self):
        self._request(None)
        self._request(self.size(-2))
        session = session_manager.get_session(self.session_id)
        dress_hash = tryon.get_dress_hash(Image.fromarray(self.capture))[:16]
        self.assertNotEqual(session.get_last_good_render(dress_hash), session.get_size_render(self.size(-2), dress_hash))


if __name__ == "__main__":
    unittest.main()
