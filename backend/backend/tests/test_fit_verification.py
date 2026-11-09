"""
Cross-size verification, calibrated on real renders (2026-09-28): garment width only flags gross errors (render-to-render
noise ~6% of shoulder width); near-identical sizes need both a close silhouette and a close torso appearance.
"""
import unittest

from app.services.silhouette_verifier import (
    FIT_DIRECTION_NOISE,
    NEAR_IDENTICAL_APPEARANCE,
    WONT_FIT_MAX_WIDENING,
    compare_sizes,
    compare_to_recommended,
    fit_family,
)


def m(torso, outer, appearance=None):
    return {"torso_width": torso, "outer_width": outer, "appearance": appearance}


REC = m(0.90, 1.40)


class TestDirectionAgainstRecommended(unittest.TestCase):
    def test_reports_raw_direction(self):
        self.assertTrue(compare_to_recommended(m(0.95, 1.41), REC, "relaxed")["as_expected"])
        self.assertFalse(compare_to_recommended(m(0.895, 1.395), REC, "loose")["as_expected"])
        self.assertTrue(compare_to_recommended(m(0.88, 1.38), REC, "snug")["as_expected"])

    def test_noise_level_readings_do_not_fail(self):
        """Real data: a visibly tighter S read +5.4%, the visibly most oversized XXL -0.5%: neither is an error."""
        self.assertTrue(compare_to_recommended(m(0.90 + 0.054, 1.45), REC, "snug")["passed"])
        self.assertTrue(compare_to_recommended(m(0.895, 1.3965), REC, "oversized")["passed"])

    def test_gross_errors_fail(self):
        self.assertFalse(compare_to_recommended(m(0.90 + FIT_DIRECTION_NOISE + 0.01, 1.5), REC, "tight")["passed"])
        self.assertFalse(compare_to_recommended(m(0.82, 1.32), REC, "loose")["passed"])

    def test_wont_fit_uses_a_stricter_limit(self):
        self.assertLess(WONT_FIT_MAX_WIDENING, FIT_DIRECTION_NOISE)
        d = compare_to_recommended(m(0.90 + 0.042, 1.44), REC, "does_not_fit")
        self.assertTrue(d["passed"], "not a gross error for the retry logic")
        self.assertGreater(d["torso_delta"], WONT_FIT_MAX_WIDENING, "but too wide to show as 'won't fit'")

    def test_unmeasurable_is_skipped_not_failed(self):
        self.assertIsNone(compare_to_recommended(None, REC, "loose")["passed"])
        self.assertIsNone(compare_to_recommended(m(None, None), REC, "loose")["passed"])
        self.assertIsNone(compare_to_recommended(m(0.95, 1.5), REC, "perfect")["passed"])


class TestNearIdentical(unittest.TestCase):
    def test_look_alike_pair_from_real_run(self):
        """Lean L vs XL looked the same: silhouette 0.039, appearance 0.232."""
        a, b = m(0.94, 1.448, [0.0] * 4), m(0.91, 1.4865, [0.232] * 4)
        self.assertTrue(compare_sizes(a, "relaxed", b, "loose")["near_identical"])

    def test_same_silhouette_different_look_is_not_identical(self):
        """Real run: an open-front XS and an XXL had silhouette distance 0.005 but look nothing alike (0.64)."""
        a, b = m(0.8965, 1.4383, [0.0] * 4), m(0.8943, 1.4429, [0.643] * 4)
        self.assertFalse(compare_sizes(a, "does_not_fit", b, "oversized")["near_identical"])

    def test_threshold(self):
        base = m(0.90, 1.40, [0.0] * 4)
        self.assertTrue(compare_sizes(base, "snug", m(0.90, 1.40, [NEAR_IDENTICAL_APPEARANCE - 0.01] * 4), "tight")["near_identical"])
        self.assertFalse(compare_sizes(base, "snug", m(0.90, 1.40, [NEAR_IDENTICAL_APPEARANCE + 0.01] * 4), "tight")["near_identical"])

    def test_without_appearance_not_judged(self):
        self.assertIsNone(compare_sizes(m(0.9, 1.4), "relaxed", m(0.9, 1.4), "loose")["near_identical"])

    def test_two_wont_fit_sizes_are_the_same_state(self):
        a = m(0.88, 1.38, [0.0] * 4)
        self.assertIsNone(compare_sizes(a, "does_not_fit", a, "does_not_fit")["near_identical"])


class TestFitFamily(unittest.TestCase):
    def test_families(self):
        for q in ("relaxed", "loose", "oversized"):
            self.assertEqual(fit_family(q), "relaxed")
        for q in ("snug", "tight", "does_not_fit"):
            self.assertEqual(fit_family(q), "fitted")
        self.assertEqual(fit_family("perfect"), "standard")
        self.assertEqual(fit_family(None, "relaxed"), "relaxed")


if __name__ == "__main__":
    unittest.main()
