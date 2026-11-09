import unittest
from PIL import Image, ImageDraw
from app.services.body_profile import BodyProfileService, FrameLandmarks


def create_synthetic_person_image(
    width=300, height=500, shoulder_w=80, hip_w=65, y_offset=50, person_h=400
) -> Image.Image:
    """Generates a synthetic white-backdrop person silhouette for deterministic testing."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    center_x = width // 2
    head_top = y_offset
    head_bottom = y_offset + int(person_h * 0.1125)
    torso_top = y_offset + int(person_h * 0.1375)
    torso_bottom = y_offset + int(person_h * 0.55)
    legs_bottom = y_offset + person_h

    # Head
    draw.ellipse([center_x - 20, head_top, center_x + 20, head_bottom], fill=(200, 160, 130))
    # Torso
    sh_half = shoulder_w // 2
    hip_half = hip_w // 2
    draw.polygon([
        (center_x - sh_half, torso_top),
        (center_x + sh_half, torso_top),
        (center_x + hip_half, torso_bottom),
        (center_x - hip_half, torso_bottom)
    ], fill=(50, 70, 120))
    # Legs
    draw.rectangle([center_x - hip_half, torso_bottom, center_x - 5, legs_bottom], fill=(40, 40, 60))
    draw.rectangle([center_x + 5, torso_bottom, center_x + hip_half, legs_bottom], fill=(40, 40, 60))

    return img



class TestBodyProfileService(unittest.TestCase):
    def setUp(self):
        self.service = BodyProfileService()

    def test_single_frame_extraction(self):
        """Verify single frame extraction returns valid FrameLandmarks."""
        img = create_synthetic_person_image(shoulder_w=90, hip_w=70)
        lm = self.service.extract_single_frame(img)
        self.assertIsNotNone(lm)
        self.assertGreater(lm.shoulder_width, 0.1)
        self.assertGreater(lm.height_ratio, 0.4)

    def test_single_capture_profile(self):
        """A lone front-on capture (no hold frames) yields a complete profile."""
        profile = self.service.extract_profile(create_synthetic_person_image(shoulder_w=86, hip_w=69))
        self.assertIsNotNone(profile)
        self.assertEqual(profile.frame_count_sampled, 1)
        self.assertGreater(profile.shoulder_to_hip_ratio, 1.0)
        self.assertIn(profile.body_type, [
            "Athletic / V-Shape", "Hourglass / Curved", "Rectangle / Balanced",
            "Pear / Triangle", "Inverted Triangle"
        ])

    def test_hold_frames_median_rejects_jittery_frame(self):
        """One jittery hold frame must not move the measurement: the median of 3 frames discards it."""
        baseline = self.service.extract_profile(create_synthetic_person_image(shoulder_w=86, hip_w=69))
        profile = self.service.extract_profile(
            create_synthetic_person_image(shoulder_w=86, hip_w=69),
            [
                create_synthetic_person_image(shoulder_w=86, hip_w=69),
                create_synthetic_person_image(shoulder_w=130, hip_w=100),  # jitter outlier
            ],
        )
        self.assertEqual(profile.frame_count_sampled, 3)
        self.assertAlmostEqual(profile.shoulder_width_ratio, baseline.shoulder_width_ratio, places=3)
        self.assertAlmostEqual(profile.chest_proportion, baseline.chest_proportion, places=3)

    def test_depth_uses_population_ratio(self):
        """Sagittal depth is always the fixed population ratio of shoulder width."""
        from app.services.body_profile import SAGITTAL_DEPTH_TO_SHOULDER_RATIO
        self.assertEqual(SAGITTAL_DEPTH_TO_SHOULDER_RATIO, 0.72)
        profile = self.service.extract_profile(create_synthetic_person_image(shoulder_w=86, hip_w=69))
        summary = profile.keypoints_summary
        self.assertAlmostEqual(
            summary["sagittal_depth_est"],
            round(summary["transverse_shoulder"] * SAGITTAL_DEPTH_TO_SHOULDER_RATIO, 3),
            places=3,
        )

    def test_classify_body_type(self):
        """Test body type classification heuristics."""
        # Athletic / V-Shape
        b1 = self.service._classify_body_type(sh_to_hip=1.28, chest_prop=1.1, waist_prop=0.82, hip_prop=1.0)
        self.assertEqual(b1, "Athletic / V-Shape")

        # Pear / Triangle
        b2 = self.service._classify_body_type(sh_to_hip=0.90, chest_prop=0.95, waist_prop=0.92, hip_prop=1.1)
        self.assertEqual(b2, "Pear / Triangle")

        # Inverted Triangle
        b3 = self.service._classify_body_type(sh_to_hip=1.22, chest_prop=1.05, waist_prop=0.95, hip_prop=1.0)
        self.assertEqual(b3, "Inverted Triangle")

    def test_generate_face_and_hands_mask(self):
        """Verify binary and soft-feathered mask generation for face and hands."""
        import numpy as np
        img = create_synthetic_person_image(width=300, height=500)
        binary_mask, feathered_mask = self.service.generate_face_and_hands_mask(img, feather_radius=15)

        self.assertEqual(binary_mask.shape, (500, 300))
        self.assertEqual(feathered_mask.shape, (500, 300))
        self.assertEqual(binary_mask.dtype, np.uint8)

        # Protected face region in upper portion must have marked pixels
        self.assertTrue(np.any(binary_mask[40:110, 120:180] == 255), "Face region must be protected")
        # Lower garment/torso region must be editable (0)
        self.assertTrue(np.any(binary_mask[180:240, 120:180] == 0), "Garment region must be editable")

        # Feathered mask must be float in range [0.0, 1.0] with soft gradient
        self.assertGreaterEqual(feathered_mask.min(), 0.0)
        self.assertLessEqual(feathered_mask.max(), 1.0)
        # Gradient exists: intermediate values between 0.1 and 0.9 exist around edges
        gradient_pixels = np.logical_and(feathered_mask > 0.1, feathered_mask < 0.9)
        self.assertTrue(np.any(gradient_pixels), "Mask must have soft feathered boundary gradient")

    def test_calibrated_distance_and_measurements(self):
        """Verify that 5-6m capture calibration produces realistic metric measurements."""
        profile = self.service.extract_profile(
            create_synthetic_person_image(shoulder_w=40, hip_w=32, person_h=180),
            [create_synthetic_person_image(shoulder_w=42, hip_w=33, person_h=180)],
        )
        self.assertIsNotNone(profile)
        summary = profile.keypoints_summary
        self.assertIn("capture_distance_range_meters", summary)
        self.assertEqual(summary["capture_distance_range_meters"], [2.0, 3.0])
        self.assertIn("calibrated_scale_factor", summary)
        self.assertIn("estimated_measurements_cm", summary)

        measurements = summary["estimated_measurements_cm"]
        self.assertIn("height", measurements)
        self.assertIn("shoulder_width", measurements)
        self.assertIn("chest_circumference", measurements)
        self.assertIn("waist_circumference", measurements)
        self.assertIn("hip_circumference", measurements)

        self.assertGreater(measurements["height"], 100.0)
        self.assertLess(measurements["height"], 250.0)
        self.assertGreater(measurements["shoulder_width"], 20.0)
        self.assertGreater(measurements["chest_circumference"], 40.0)


if __name__ == "__main__":
    unittest.main()

