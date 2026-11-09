import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.services import silhouette_verifier as verifier_module
from app.services.body_profile import BodyProfileService, FrameLandmarks, classify_torso_shape
from app.services.silhouette_verifier import MIDSECTION_MAX_LOSS, SilhouetteVerifier
from app.services.size_chart import recommend_size

W, H = 800, 1000
CX = 400
ARM_W = 60
SHOULDER_HALF = 130  # skeletal: does not change with body volume
SH_Y, ELBOW_Y = 0.25, 0.45


def make_body(torso_half=100, arm_gap=0, mask=True) -> FrameLandmarks:
    """
    Synthetic front capture, arms hanging at the sides. The torso is 2*torso_half px wide between fixed shoulder
    joints; each arm is a 60px band hanging from the shoulder to an elbow arm_gap px outside the torso
    (0 = touching, the normal capture pose, so the arms merge into the silhouette run).
    """
    elbow_off = torso_half + arm_gap + ARM_W // 2
    seg = None
    if mask:
        seg = np.zeros((H, W), dtype=np.float32)
        seg[250:560, CX - torso_half:CX + torso_half] = 1.0
        for row in range(250, 700):
            t = min(1.0, (row / (H - 1) - SH_Y) / (ELBOW_Y - SH_Y))
            off = SHOULDER_HALF + t * (elbow_off - SHOULDER_HALF)
            for side in (-1, 1):
                c = int(round(CX + side * off))
                seg[row, c - ARM_W // 2:c + ARM_W // 2] = 1.0

    def pt(x_px, y):
        return (x_px / (W - 1), y, 0.0)

    kp = {
        "left_shoulder": pt(CX - SHOULDER_HALF, SH_Y), "right_shoulder": pt(CX + SHOULDER_HALF, SH_Y),
        "left_elbow": pt(CX - elbow_off, ELBOW_Y), "right_elbow": pt(CX + elbow_off, ELBOW_Y),
        "left_wrist": pt(CX - elbow_off, 0.62), "right_wrist": pt(CX + elbow_off, 0.62),
        "left_hip": pt(CX - 60, 0.55), "right_hip": pt(CX + 60, 0.55),
    }
    return FrameLandmarks(
        height_ratio=0.8, shoulder_width=2 * SHOULDER_HALF / W, torso_length=0.3, hip_width=120 / W,
        leg_length=0.4, yaw_angle=0.0, confidence=0.95, keypoints=kp, segmentation_mask=seg, image_size=(W, H),
    )


def expected_fullness(torso_half=100):
    return torso_half / SHOULDER_HALF


class TestTorsoFullness(unittest.TestCase):
    def setUp(self):
        self.service = BodyProfileService()

    def test_arms_touching_torso_are_excluded(self):
        """The normal capture pose: arms merged into the silhouette are measured out, not dropped."""
        value = self.service.torso_fullness(make_body(torso_half=100))
        self.assertIsNotNone(value)
        self.assertAlmostEqual(value, expected_fullness(100), delta=0.02)

    def test_arms_apart_measure_torso_directly(self):
        value = self.service.torso_fullness(make_body(torso_half=100, arm_gap=25))
        self.assertAlmostEqual(value, expected_fullness(100), delta=0.02)

    def test_wider_midsection_scores_fuller(self):
        lean = self.service.torso_fullness(make_body(torso_half=80))
        full = self.service.torso_fullness(make_body(torso_half=150))
        self.assertGreater(full, lean * 1.2)

    def test_unmeasurable_without_mask(self):
        self.assertIsNone(self.service.torso_fullness(make_body(mask=False)))

    def test_classify_torso_shape_bands(self):
        self.assertIsNone(classify_torso_shape(None))
        self.assertEqual(classify_torso_shape(0.90), "lean")
        self.assertEqual(classify_torso_shape(1.00), "average")
        self.assertEqual(classify_torso_shape(1.10), "full")
        self.assertEqual(classify_torso_shape(1.30), "very_full")

    def test_profile_carries_fullness(self):
        profile = self.service.profile_from_frames([make_body(torso_half=100)] * 3)
        self.assertAlmostEqual(profile.torso_fullness, expected_fullness(100), delta=0.02)
        self.assertEqual(profile.torso_shape, classify_torso_shape(profile.torso_fullness))
        self.assertEqual(profile.keypoints_summary["torso_fullness_frames"], 3)

    def test_fullness_never_changes_size_recommendation(self):
        """Release gate: torso fullness feeds the prompt and verifier only, never XS-XXL matching."""
        profile = self.service.profile_from_frames([make_body(torso_half=100)])
        for category in (None, "shirt", "blazer", "dress", "kurti"):
            baseline = recommend_size(profile, garment_category=category).recommended_size
            for fullness, shape in ((0.8, "lean"), (1.3, "very_full")):
                varied = profile.model_copy(update={"torso_fullness": fullness, "torso_shape": shape})
                self.assertEqual(recommend_size(varied, garment_category=category).recommended_size, baseline)


class TestMidsectionVerification(unittest.TestCase):
    def setUp(self):
        self.verifier = SilhouetteVerifier()

    def test_flattened_render_fails(self):
        result = self.verifier.check_midsection(make_body(torso_half=150), make_body(torso_half=110))
        self.assertFalse(result["preserved"])
        self.assertLess(result["change"], -MIDSECTION_MAX_LOSS)

    def test_same_or_wider_render_passes(self):
        self.assertTrue(self.verifier.check_midsection(make_body(150), make_body(150))["preserved"])
        self.assertTrue(self.verifier.check_midsection(make_body(110), make_body(150))["preserved"])

    def test_skipped_when_unmeasurable(self):
        self.assertIsNone(self.verifier.check_midsection(make_body(150), make_body(150, mask=False)))
        self.assertIsNone(self.verifier.check_midsection(None, make_body(150)))

    def test_verify_fails_flattened_render_despite_good_skeleton_score(self):
        """Whole-body IoU and skeletal ratios barely move when only the stomach is slimmed; the midsection gate must."""
        capture_img, render_img = Image.new("RGB", (W, H), "white"), Image.new("RGB", (W, H), "white")
        frames = {id(capture_img): make_body(torso_half=150), id(render_img): make_body(torso_half=110)}
        with patch.object(verifier_module.body_profile_service, "extract_single_frame", side_effect=lambda im: frames[id(im)]):
            result = self.verifier.verify(user_img=capture_img, generated_img=render_img)
        self.assertFalse(result.passed)
        self.assertFalse(result.details["midsection_preserved"])
        self.assertLess(result.details["midsection_change"], -MIDSECTION_MAX_LOSS)

        frames[id(render_img)] = make_body(torso_half=150)
        with patch.object(verifier_module.body_profile_service, "extract_single_frame", side_effect=lambda im: frames[id(im)]):
            result = self.verifier.verify(user_img=capture_img, generated_img=render_img)
        self.assertTrue(result.details["midsection_preserved"])


if __name__ == "__main__":
    unittest.main()
