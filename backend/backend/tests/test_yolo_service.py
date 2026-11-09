import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from app.services.yolox_detector import YoloxDetector as YOLO

from app.services.yolo_service import (
    PERSON_CLASS_ID,
    PERSON_CLASS_NAME,
    PersonDetection,
    YOLODetectionResult,
    YOLOService,
    get_yolo_service,
)


class TestYOLOService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize service once for the test suite
        cls.service = YOLOService()

    def test_service_initialization(self):
        """Verify service initializes with expected configurations and defaults."""
        self.assertIsNotNone(self.service.model)
        self.assertIsInstance(self.service.model, YOLO)
        self.assertEqual(self.service.device, "cpu")
        self.assertGreaterEqual(self.service.confidence, 0.0)
        self.assertLessEqual(self.service.confidence, 1.0)
        self.assertGreaterEqual(self.service.iou, 0.0)
        self.assertLessEqual(self.service.iou, 1.0)

    def test_model_loading(self):
        """Verify the YOLO model loads without error and is accessible."""
        model = self.service.model
        self.assertTrue(hasattr(model, "predict"))

    def test_model_reuse(self):
        """Verify model weights are loaded ONCE and reused across service instances."""
        service_a = YOLOService()
        service_b = YOLOService()
        self.assertIs(
            service_a.model,
            service_b.model,
            "Service instances with matching parameters must reuse the cached YOLO model",
        )
        # Global getter should also reuse the singleton instance
        global_service = get_yolo_service()
        self.assertIsInstance(global_service, YOLOService)

    def test_zero_person_detection_handling(self):
        """Verify handling when an image has zero people (blank canvas)."""
        blank_image = np.zeros((320, 320, 3), dtype=np.uint8)
        result = self.service.detect_person(blank_image)

        self.assertIsInstance(result, YOLODetectionResult)
        self.assertEqual(result.count, 0)
        self.assertEqual(len(result.detections), 0)
        self.assertIsNone(result.primary_person)
        self.assertEqual(result.image_width, 320)
        self.assertEqual(result.image_height, 320)

        # Convenience helper
        primary = self.service.get_primary_person(blank_image)
        self.assertIsNone(primary)

        all_people = self.service.get_all_people(blank_image)
        self.assertEqual(all_people, [])

    def test_inference_on_valid_pil_and_numpy_images(self):
        """Verify inference accepts both PIL Image and NumPy ndarray without mutating input."""
        # PIL Image
        pil_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        result_pil = self.service.detect_person(pil_img)
        self.assertIsInstance(result_pil, YOLODetectionResult)
        self.assertEqual(result_pil.image_width, 256)
        self.assertEqual(result_pil.image_height, 256)

        # NumPy ndarray
        np_img = np.full((240, 320, 3), 128, dtype=np.uint8)
        original_copy = np_img.copy()
        result_np = self.service.detect_person(np_img)
        self.assertIsInstance(result_np, YOLODetectionResult)
        self.assertEqual(result_np.image_width, 320)
        self.assertEqual(result_np.image_height, 240)

        # Strict check: image array was NOT modified (no drawn rectangles, text, or altered pixels)
        self.assertTrue(
            np.array_equal(np_img, original_copy),
            "Input image was modified by inference! Detection must be strictly non-mutating.",
        )

    def test_bounding_box_validity_and_clamping(self):
        """Verify bounding box clamping strictly bounds within [0, W] and [0, H]."""
        w, h = 640, 480
        # Out-of-bounds coordinates
        x1, y1, x2, y2 = self.service.clamp_bbox(-50.4, -10.2, 700.8, 500.1, w, h)
        self.assertEqual((x1, y1, x2, y2), (0, 0, 640, 480))

        # Inverted coordinates
        ix1, iy1, ix2, iy2 = self.service.clamp_bbox(300, 400, 100, 200, w, h)
        self.assertEqual((ix1, iy1, ix2, iy2), (100, 200, 300, 400))

        # Validity check
        self.assertTrue(self.service.is_valid_bbox(10, 10, 50, 50))
        self.assertFalse(self.service.is_valid_bbox(50, 50, 10, 10))
        self.assertFalse(self.service.is_valid_bbox(10, 10, 10, 50))  # zero width
        self.assertFalse(self.service.is_valid_bbox(10, 10, 50, 10))  # zero height
        self.assertFalse(self.service.is_valid_bbox(10, 10, 11, 11, min_area=10))

    def test_confidence_range_and_custom_thresholds(self):
        """Verify confidence values are strictly within [0, 1]."""
        det = PersonDetection(x1=10, y1=10, x2=50, y2=100, confidence=0.8876, area=3600)
        self.assertGreaterEqual(det.confidence, 0.0)
        self.assertLessEqual(det.confidence, 1.0)
        d_dict = det.to_dict()
        self.assertEqual(d_dict["confidence"], 0.8876)
        self.assertEqual(d_dict["width"], 40)
        self.assertEqual(d_dict["height"], 90)

    def test_primary_person_selection_deterministic(self):
        """
        Verify primary person selection uses deterministic prominence ranking:
        prominence = area * (confidence ** 2).
        """
        # Case: Foreground person with large area vs background person with slightly higher confidence
        bg_person = PersonDetection(
            x1=500, y1=100, x2=550, y2=200, confidence=0.95, area=5000  # score = 5000 * 0.9025 = 4512
        )
        fg_person = PersonDetection(
            x1=150, y1=50, x2=450, y2=450, confidence=0.85, area=120000  # score = 120000 * 0.7225 = 86700
        )
        side_person = PersonDetection(
            x1=20, y1=80, x2=120, y2=300, confidence=0.70, area=22000  # score = 22000 * 0.49 = 10780
        )

        detections = [bg_person, fg_person, side_person]
        selected = self.service.select_primary_person(detections)
        self.assertIsNotNone(selected)
        self.assertIs(selected, fg_person, "Foreground person with significant area must be selected as primary")

        # Reverse input order to ensure order-independence (determinism)
        selected_rev = self.service.select_primary_person([fg_person, bg_person, side_person])
        self.assertIs(selected_rev, fg_person)

        # Empty list handling
        self.assertIsNone(self.service.select_primary_person([]))

    def test_mocked_multiple_person_inference(self):
        """Verify parsing and structure when multiple people are returned by YOLO."""
        service = YOLOService()

        # Mock predict output simulating 2 detected people
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 2

        # xyxy: [x1, y1, x2, y2]
        mock_boxes.xyxy = np.array([
            [50.0, 50.0, 150.0, 250.0],
            [200.0, 80.0, 400.0, 480.0],
        ])
        mock_boxes.conf = np.array([0.80, 0.90])
        mock_boxes.cls = np.array([PERSON_CLASS_ID, PERSON_CLASS_ID])

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        with patch.object(service.model, "predict", return_value=[mock_result]):
            sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
            res = service.detect_person(sample_img)

            self.assertEqual(res.count, 2)
            self.assertEqual(len(res.detections), 2)
            self.assertIsNotNone(res.primary_person)
            # The second person is larger (200x400 = 80000) and higher conf (0.90)
            self.assertEqual(res.primary_person.x1, 200)
            self.assertEqual(res.primary_person.x2, 400)
            self.assertEqual(res.primary_person.confidence, 0.90)

    def test_invalid_image_inputs(self):
        """Verify error handling on invalid image inputs."""
        with self.assertRaises(ValueError):
            self.service.detect_person(None)

        with self.assertRaises(TypeError):
            self.service.detect_person("invalid_path_or_string")

        with self.assertRaises(ValueError):
            # 1D array is invalid
            self.service.detect_person(np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
