import asyncio
import base64
import csv
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from PIL import Image

from app.models import MeasurementCorrectionRequest, ProfileCaptureRequest
from app.routers import tryon
from app.services import body_profile
from app.services.body_profile import BodyProfileService, FrameLandmarks
from app.services.session_manager import session_manager

IMG_W, IMG_H = 800, 1000


def make_frame(shoulder_half=0.11, ankle_vis=0.9, ankle_y=0.84, feet_cut=False, mask=True) -> FrameLandmarks:
    """
    Synthetic 800x1000 capture. With the patched calibration below, 1px = 0.25cm:
    silhouette rows 150..849 (700px = 175cm), torso 200px wide (50cm), shoulders 0.22 × 800px (44cm).
    """
    seg = None
    if mask:
        seg = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        seg[150:IMG_H if feet_cut else 850, 350:450] = 1.0  # head, legs
        seg[250:500, 300:500] = 1.0  # torso
    kp = {
        "left_shoulder": (0.5 - shoulder_half, 0.25, 0.0),
        "right_shoulder": (0.5 + shoulder_half, 0.25, 0.0),
        "left_hip": (0.44, 0.5, 0.0),
        "right_hip": (0.56, 0.5, 0.0),
        "left_ankle": (0.47, ankle_y, 0.0),
        "right_ankle": (0.53, ankle_y, 0.0),
    }
    vis = {"left_ankle": ankle_vis, "right_ankle": ankle_vis}
    return FrameLandmarks(
        height_ratio=0.7, shoulder_width=2 * shoulder_half, torso_length=0.25, hip_width=0.12,
        leg_length=0.34, yaw_angle=0.0, confidence=0.95, keypoints=kp,
        segmentation_mask=seg, visibility=vis, image_size=(IMG_W, IMG_H),
    )


# Focal length 1080px at 1080px reference height, 2.5m → 1000px at this capture height → 0.25 cm/px
CALIBRATION = [
    patch.object(body_profile, "MEASUREMENT_FOCAL_LENGTH_PX", 1080.0),
    patch.object(body_profile, "MEASUREMENT_REFERENCE_HEIGHT_PX", 1080),
    patch.object(body_profile, "MEASUREMENT_CALIBRATION_DISTANCE_M", 2.5),
]


class TestBodyMeasurements(unittest.TestCase):
    def setUp(self):
        self.service = BodyProfileService()
        for p in CALIBRATION:
            p.start()
            self.addCleanup(p.stop)

    def test_full_body_capture(self):
        m = self.service.measure_body([make_frame()])
        self.assertAlmostEqual(m.height.value_cm, 175.0, places=1)
        self.assertEqual(m.height.source, "silhouette")
        self.assertAlmostEqual(m.shoulder_width.value_cm, 44.0, places=1)
        self.assertEqual(m.shoulder_width.source, "landmarks")
        self.assertAlmostEqual(m.chest_width.value_cm, 50.0, places=1)
        self.assertAlmostEqual(m.waist_width.value_cm, 50.0, places=1)
        self.assertEqual(m.assumed_distance_m, 2.5)
        for field in (m.height, m.shoulder_width, m.chest_width, m.waist_width):
            self.assertEqual(field.confidence, "estimated")

    def test_hidden_ankles_fall_back_to_shoulder_ratio(self):
        m = self.service.measure_body([make_frame(ankle_vis=0.1)])
        self.assertEqual(m.height.source, "shoulder_ratio")
        self.assertAlmostEqual(m.height.value_cm, round(44.0 / body_profile.SHOULDER_WIDTH_TO_HEIGHT_RATIO, 1), places=1)

    def test_extrapolated_offscreen_ankles_are_rejected(self):
        """MediaPipe can report high visibility for an ankle it placed below the frame edge."""
        m = self.service.measure_body([make_frame(ankle_y=1.15)])
        self.assertEqual(m.height.source, "shoulder_ratio")

    def test_feet_cut_by_frame_edge_falls_back(self):
        m = self.service.measure_body([make_frame(feet_cut=True)])
        self.assertEqual(m.height.source, "shoulder_ratio")

    def test_median_across_capture_frames(self):
        frames = [make_frame(shoulder_half=0.11), make_frame(shoulder_half=0.11), make_frame(shoulder_half=0.20)]
        m = self.service.measure_body(frames)
        self.assertAlmostEqual(m.shoulder_width.value_cm, 44.0, places=1)

    def test_arms_merged_into_silhouette_drop_chest_and_waist(self):
        """Arms against the torso widen the silhouette well past the shoulders; show nothing rather than arm-to-arm width."""
        frame = make_frame()
        frame.segmentation_mask[250:500, 220:580] = 1.0  # torso + arms: 360px = 90cm vs 44cm shoulders
        m = self.service.measure_body([frame])
        self.assertIsNone(m.chest_width)
        self.assertIsNone(m.waist_width)
        self.assertAlmostEqual(m.shoulder_width.value_cm, 44.0, places=1)

    def test_no_mask_gives_shoulder_only(self):
        m = self.service.measure_body([make_frame(mask=False)])
        self.assertAlmostEqual(m.shoulder_width.value_cm, 44.0, places=1)
        self.assertEqual(m.height.source, "shoulder_ratio")
        self.assertIsNone(m.chest_width)
        self.assertIsNone(m.waist_width)

    def test_contour_fallback_frame_yields_no_measurements(self):
        fallback = self.service._geometric_contour_fallback(np.full((400, 300, 3), 255, dtype=np.uint8))
        m = self.service.measure_body([fallback])
        self.assertIsNone(m.height)
        self.assertIsNone(m.shoulder_width)

    def test_measurements_do_not_change_sizing_profile(self):
        """extract_profile must equal profile_from_frames on the same frames: measurements are additive only."""
        img = Image.new("RGB", (300, 500), (255, 255, 255))
        frames = self.service.extract_frames(img)
        self.assertEqual(self.service.extract_profile(img), self.service.profile_from_frames(frames))


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestMeasurementEndpoints(unittest.TestCase):
    def setUp(self):
        for p in CALIBRATION:
            p.start()
            self.addCleanup(p.stop)

    def test_extract_profile_returns_measurements_separately(self):
        sid = "sess_measure_endpoint"
        self.addCleanup(session_manager.purge_session, sid)
        with patch.object(tryon.body_profile_service, "extract_frames", return_value=[make_frame()]):
            resp = asyncio.run(tryon.extract_profile(ProfileCaptureRequest(
                image=_b64(Image.new("RGB", (80, 100))), session_id=sid,
            )))
        self.assertAlmostEqual(resp["body_measurements"].height.value_cm, 175.0, places=1)
        self.assertNotIn("height_cm", resp["body_profile"].model_dump())
        stored = session_manager.get_session(sid).get_body_measurements()
        self.assertAlmostEqual(stored.shoulder_width.value_cm, 44.0, places=1)

    def test_measurement_failure_does_not_block_profile(self):
        with patch.object(tryon.body_profile_service, "extract_frames", return_value=[make_frame()]), \
             patch.object(tryon.body_profile_service, "measure_body", side_effect=RuntimeError("boom")):
            resp = asyncio.run(tryon.extract_profile(ProfileCaptureRequest(image=_b64(Image.new("RGB", (80, 100))))))
        self.assertTrue(resp["success"])
        self.assertIsNotNone(resp["body_profile"])
        self.assertIsNone(resp["body_measurements"])

    def test_correction_is_logged_against_stored_estimate(self):
        sid = "sess_measure_correction"
        self.addCleanup(session_manager.purge_session, sid)
        session_manager.get_or_create_session(sid).set_body_measurements(self._service_measure())
        with patch("builtins.print") as mock_print:
            resp = asyncio.run(tryon.measurement_correction(MeasurementCorrectionRequest(
                session_id=sid, field="height", displayed_cm=175.0, corrected_cm=168.0,
            )))
        self.assertTrue(resp["success"])
        logged = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("system_estimate_cm=175.0", logged)
        self.assertIn("corrected_cm=168.0", logged)

    def test_correction_without_session_measurements_is_404(self):
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(tryon.measurement_correction(MeasurementCorrectionRequest(
                session_id="sess_missing", field="height", displayed_cm=175.0, corrected_cm=170.0,
            )))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_purge_erases_measurements(self):
        sid = "sess_measure_purge"
        session_manager.get_or_create_session(sid).set_body_measurements(self._service_measure())
        session_manager.purge_session(sid)
        self.assertIsNone(session_manager.get_session(sid))

    def _correct(self, sid: str, field: str, displayed: float, corrected: float):
        with patch("builtins.print"):
            return asyncio.run(tryon.measurement_correction(MeasurementCorrectionRequest(
                session_id=sid, field=field, displayed_cm=displayed, corrected_cm=corrected,
            )))

    def test_correction_log_file_not_written_when_disabled(self):
        sid = "sess_measure_log_off"
        self.addCleanup(session_manager.purge_session, sid)
        session_manager.get_or_create_session(sid).set_body_measurements(self._service_measure())
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "logs", "measurement_corrections.csv")
            with patch.object(tryon, "MEASUREMENT_CORRECTION_LOG", False),                  patch.object(tryon, "MEASUREMENT_CORRECTION_LOG_PATH", log_path):
                self._correct(sid, "height", 175.0, 168.0)
            self.assertFalse(os.path.exists(log_path))

    def test_correction_log_file_appends_rows_with_single_header(self):
        sid = "sess_measure_log_on"
        self.addCleanup(session_manager.purge_session, sid)
        session_manager.get_or_create_session(sid).set_body_measurements(self._service_measure())
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "logs", "measurement_corrections.csv")
            with patch.object(tryon, "MEASUREMENT_CORRECTION_LOG", True), \
                 patch.object(tryon, "MEASUREMENT_CORRECTION_LOG_PATH", log_path):
                self._correct(sid, "height", 175.0, 168.0)
                self._correct(sid, "shoulder_width", 42.0, 45.0)
            with open(log_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["session_id"], sid)
        self.assertEqual(rows[0]["field"], "height")
        self.assertEqual(float(rows[0]["system_estimate_cm"]), 175.0)
        self.assertEqual(rows[0]["height_source"], "silhouette")
        self.assertEqual(float(rows[0]["corrected_cm"]), 168.0)
        self.assertEqual(rows[1]["field"], "shoulder_width")
        self.assertEqual(float(rows[1]["system_estimate_cm"]), 44.0)
        self.assertEqual(float(rows[1]["displayed_cm"]), 42.0)

    def test_calibration_capture_saved_only_when_enabled(self):
        request = ProfileCaptureRequest(image=_b64(Image.new("RGB", (80, 100), (10, 20, 30))))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(tryon, "CALIBRATION_CAPTURE_DIR", tmp), \
             patch.object(tryon.body_profile_service, "extract_frames", return_value=[make_frame()]), \
             patch("builtins.print"):
            with patch.object(tryon, "CALIBRATION_CAPTURE_SAVE", False):
                asyncio.run(tryon.extract_profile(request))
            self.assertEqual(os.listdir(tmp), [])

            with patch.object(tryon, "CALIBRATION_CAPTURE_SAVE", True):
                asyncio.run(tryon.extract_profile(request))
            saved = os.listdir(tmp)
            self.assertEqual(len(saved), 1)
            with Image.open(os.path.join(tmp, saved[0])) as img:
                self.assertEqual(img.size, (80, 100))
                self.assertEqual(img.convert("RGB").getpixel((0, 0)), (10, 20, 30))

    @staticmethod
    def _service_measure():
        return BodyProfileService().measure_body([make_frame()])


if __name__ == "__main__":
    unittest.main()
