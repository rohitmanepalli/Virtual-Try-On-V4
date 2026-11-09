import unittest
from PIL import Image, ImageDraw
from app.models import BodyProfile
from app.services.silhouette_verifier import SilhouetteVerifier


def make_test_silhouette(w=300, h=500, shoulder_w=80, hip_w=65, y_offset=0, warp_ratio=1.0) -> Image.Image:
    """Creates a test silhouette image with controllable shoulder width, offset, and warping."""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx = w // 2

    # Head
    draw.ellipse([cx - 20, 50 + y_offset, cx + 20, 95 + y_offset], fill=(180, 140, 110))
    # Torso
    sh_half = int((shoulder_w * warp_ratio) // 2)
    hip_half = hip_w // 2
    draw.polygon([
        (cx - sh_half, 105 + y_offset),
        (cx + sh_half, 105 + y_offset),
        (cx + hip_half, 270 + y_offset),
        (cx - hip_half, 270 + y_offset)
    ], fill=(60, 80, 130))
    # Legs
    draw.rectangle([cx - hip_half, 270 + y_offset, cx - 5, 450 + y_offset], fill=(40, 40, 60))
    draw.rectangle([cx + 5, 270 + y_offset, cx + hip_half, 450 + y_offset], fill=(40, 40, 60))

    return img


class TestSilhouetteCalibration(unittest.TestCase):
    """
    Validation test suite for empirical silhouette verification threshold calibration.
    Evaluates scoring across genuine proportion-locked renders vs warped/distorted renders.
    """

    def setUp(self):
        self.verifier = SilhouetteVerifier()
        self.sample_profile = BodyProfile(
            height_ratio=0.80,
            shoulder_width_ratio=0.27,
            torso_aspect_ratio=1.15,
            shoulder_to_hip_ratio=1.23,
            chest_proportion=1.05,
            waist_proportion=0.85,
            hip_proportion=1.0,
            body_type="Athletic / V-Shape",
            confidence_score=0.9,
            frame_count_sampled=6
        )

    def test_genuine_render_passes_provisional_threshold(self):
        """A well-proportioned render should comfortably exceed the provisional 0.70 threshold."""
        user_img = make_test_silhouette(shoulder_w=80, hip_w=65)
        # Genuine render has the same body with dress overlay
        genuine_render = make_test_silhouette(shoulder_w=82, hip_w=66)

        result = self.verifier.verify(user_img, genuine_render, body_profile=self.sample_profile)
        self.assertTrue(result.passed, f"Genuine render should pass. Score: {result.score}")
        self.assertGreaterEqual(result.score, 0.70)

    def test_severely_distorted_render_fails_threshold(self):
        """A render with warped/bloated shoulders or collapsed anatomy must fail."""
        user_img = make_test_silhouette(shoulder_w=80, hip_w=65)
        # Render has 1.8x bloated shoulders and displaced torso (severe distortion)
        warped_render = make_test_silhouette(shoulder_w=80, hip_w=65, warp_ratio=1.8, y_offset=40)

        result = self.verifier.verify(user_img, warped_render, body_profile=self.sample_profile)
        self.assertFalse(result.passed, f"Distorted render should fail verification. Score: {result.score}")
        self.assertLess(result.score, 0.70)

    def test_threshold_sensitivity_sweep(self):
        """
        Sweeps candidate thresholds (0.60, 0.65, 0.70, 0.75, 0.80) to validate ROC discrimination
        between aligned vs warped renders.
        """
        user_img = make_test_silhouette(shoulder_w=80, hip_w=65)
        good_render = make_test_silhouette(shoulder_w=81, hip_w=65)
        bad_render = make_test_silhouette(shoulder_w=80, hip_w=65, warp_ratio=1.7)

        good_score = self.verifier.verify(user_img, good_render, body_profile=self.sample_profile).score
        bad_score = self.verifier.verify(user_img, bad_render, body_profile=self.sample_profile).score

        print(f"\n[CALIBRATION REPORT] Genuine Score: {good_score:.3f} | Distorted Score: {bad_score:.3f}")
        # Ensure a clean margin of separation (at least 0.15 gap)
        separation_margin = good_score - bad_score
        self.assertGreater(separation_margin, 0.15, "Verifier must cleanly separate good vs distorted renders.")

        # Provisional 0.70 falls cleanly between genuine and distorted scores
        self.assertGreater(good_score, 0.70)
        self.assertLess(bad_score, 0.70)


if __name__ == "__main__":
    unittest.main()
