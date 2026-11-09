"""
Face restoration quality: ONE face, the real one, with no visible mask.

History this guards against:
- The render's own face was served untouched whenever MediaPipe could see a face in it (no restoration).
- A fallback pasted capture pixels at a plain frame-scaled position / fixed ellipse, putting a second
  face beside the model's face (ghost / double face).
- A feathered band crossing the spectacle frames mixed two pairs of glasses.
- The size-variant engine cross-faded a warped and an unwarped copy inside a feathered rectangle
  (translucent double-image rectangle).
- The real face was pasted with the webcam's own colour: a dark, olive-green face on the studio-lit body
  (30 Sep - 4 Oct test renders).
"""
import ast
import base64
import io
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.services import face_preservation as fp
from app.services import gemini_client
from app.services.image_utils import finalize_result_image

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _decode(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


# ── synthetic faces with controllable geometry (stands in for MediaPipe) ────────
def ellipse_points(cx, cy, rx, ry, n=64, rot_deg=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([rx * np.cos(t), ry * np.sin(t)], 1)
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts @ R.T + [cx, cy]).astype(np.float32)


def textured(w, h, seed):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (5, 5), 0)


class SyntheticRestoreTest(unittest.TestCase):
    """Geometry/compositing contract with exact, known landmarks."""

    CAP = (300, 400)  # capture w, h
    REN = (900, 1200)  # render w, h (3x)

    def setUp(self):
        # Random-noise "faces" have no real skin at the seam or to match in colour; the seam and colour gates are
        # tested separately below
        for gate in (patch.object(fp, "SEAM_MAX_RESIDUAL", 1.0), patch.object(fp, "MAX_FACE_TONE_DELTA_E", 1e9)):
            gate.start()
            self.addCleanup(gate.stop)
        self.capture = textured(*self.CAP, seed=1)
        self.render = textured(*self.REN, seed=2)
        self.cap_pts = ellipse_points(150, 120, 51, 64)
        # the model drew the face 3x larger, shifted and slightly rotated
        self.ren_pts = ellipse_points(470, 380, 153, 192, rot_deg=4.0)
        self.points = {self.CAP: self.cap_pts, self.REN: self.ren_pts}

    def _detect(self, rgb):
        return self.points.get((rgb.shape[1], rgb.shape[0]))

    def restore(self, render=None):
        with patch.object(fp, "detect_face_points", side_effect=self._detect):
            return fp.restore_original_face(Image.fromarray(self.capture), _png(self.render if render is None else render))

    def test_full_resolution_lossless_rgb_output(self):
        out, check = self.restore()
        img = Image.open(io.BytesIO(out))
        self.assertTrue(check.restored)
        self.assertEqual(img.size, self.REN, "render canvas resolution must be kept")
        self.assertIn(img.format, ("BMP", "PNG"), "intermediate must be lossless; the single lossy encode happens at delivery")
        self.assertEqual(img.mode, "RGB", "no alpha channel")

    def test_interior_is_exactly_the_warped_capture(self):
        """
        No translucency: deep inside the face every pixel is the capture's own (Lanczos-warped) pixel, after the one
        global colour adjustment reported in the check.
        """
        out, check = self.restore()
        out = _decode(out)
        al = fp.estimate_alignment(self.cap_pts, self.ren_pts)
        ref = cv2.warpAffine(self.capture, al.M.astype(np.float64), self.REN, flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
        ref = np.clip(fp._apply_tone(ref, fp.Tone(**check.details["tone"])), 0, 255).astype(np.uint8)
        core = np.zeros(out.shape[:2], np.uint8)
        cv2.fillPoly(core, [fp._transform(ellipse_points(150, 120, 25, 32), al.M).astype(np.int32)], 1)
        diff = np.abs(out.astype(int) - ref.astype(int))[core.astype(bool)]
        # <=3 levels: OpenCV's 1/32-px fixed-point sampling rounds differently for a sub-region warp;
        # any blend with the render would differ by tens of levels
        self.assertLessEqual(int(diff.max()), 3, "interior must be the capture's pixels, not a blend")
        self.assertLess(float(diff.mean()), 0.5)
        self.assertGreater(np.abs(out.astype(int) - self.render.astype(int))[core.astype(bool)].mean(), 15)

    def test_render_untouched_outside_the_real_face(self):
        """No visible mask/rectangle: every pixel outside the real face outline is the render's own pixel."""
        out, _ = self.restore()
        out = _decode(out)
        al = fp.estimate_alignment(self.cap_pts, self.ren_pts)
        region = np.zeros(out.shape[:2], np.uint8)
        cv2.fillPoly(region, [cv2.convexHull(fp._transform(self.cap_pts, al.M).astype(np.int32))], 1)
        outside = cv2.dilate(region, np.ones((5, 5), np.uint8)) == 0
        self.assertTrue(np.array_equal(out[outside], self.render[outside]))

    def test_colour_channels_not_swapped(self):
        """RGB stays RGB end to end (a BGR slip would turn a red face blue)."""
        self.capture[:] = (200, 40, 40)
        out, _ = self.restore()
        out = _decode(out)
        r, g, b = out[380, 470]
        self.assertGreater(int(r), int(b) + 100)

    def test_single_transform_is_similarity_not_stretch(self):
        al = fp.estimate_alignment(self.cap_pts, self.ren_pts)
        self.assertTrue(al.ok, al.reason)
        a, b = al.M[0, 0], al.M[1, 0]
        c, d = al.M[0, 1], al.M[1, 1]
        self.assertAlmostEqual(float(a), float(d), places=4)
        self.assertAlmostEqual(float(b), float(-c), places=4)
        self.assertAlmostEqual(al.scale, 3.0, places=2)
        self.assertAlmostEqual(al.rotation_deg, 4.0, places=1)

    def test_invalid_transforms_are_rejected(self):
        cases = {
            "rotation": ellipse_points(470, 380, 153, 192, rot_deg=40.0),
            "stretched face": ellipse_points(470, 380, 153, 280),
        }
        for name, ren_pts in cases.items():
            with self.subTest(name):
                self.assertFalse(fp.estimate_alignment(self.cap_pts, ren_pts).ok)
        self.assertFalse(fp.estimate_alignment(None, self.ren_pts).ok)
        self.assertFalse(fp.estimate_alignment(self.cap_pts, self.ren_pts[:20]).ok)

    def test_no_face_in_render_means_no_paste_anywhere(self):
        """The old frame-mapping fallback pasted the capture face where the model had no face (double face)."""
        self.points.pop(self.REN)
        out, check = self.restore()
        self.assertFalse(check.restored)
        self.assertEqual(out, _png(self.render), "render must be returned untouched, never patched blindly")

    def test_no_face_in_capture_means_no_paste(self):
        self.points.pop(self.CAP)
        out, check = self.restore()
        self.assertFalse(check.restored)
        self.assertEqual(out, _png(self.render))

    def test_rejected_alignment_means_no_paste(self):
        self.points[self.REN] = ellipse_points(470, 380, 153, 192, rot_deg=45.0)
        out, check = self.restore()
        self.assertFalse(check.restored)
        self.assertIn("alignment rejected", check.reason)
        self.assertEqual(out, _png(self.render))

    def test_mismatched_seam_is_rejected(self):
        """A full beard-like texture in the capture meeting smooth rendered skin: the join would be visible."""
        yy, xx = np.mgrid[0:self.CAP[1], 0:self.CAP[0]]
        self.capture = np.where(((xx // 2 + yy // 3) % 2)[..., None] == 0, 30, 170).astype(np.uint8).repeat(3, 2)
        self.render = np.full((self.REN[1], self.REN[0], 3), 170, np.uint8)
        with patch.object(fp, "SEAM_MAX_RESIDUAL", 0.075):
            out, check = self.restore()
        self.assertFalse(check.restored)
        self.assertIn("seam", check.reason)
        self.assertEqual(out, _png(self.render))

    def test_fallback_mask_is_never_used(self):
        """A mask not derived from both faces' landmarks must not influence the composite."""
        with patch.object(fp, "detect_face_points", side_effect=self._detect):
            a, _ = fp.restore_original_face(Image.fromarray(self.capture), _png(self.render))
            b, _ = fp.restore_original_face(Image.fromarray(self.capture), _png(self.render),
                                            np.ones((400, 300), np.float32))
        self.assertEqual(a, b)


class LowDetailFaceRegressionTest(unittest.TestCase):
    """
    Regression (25 Sep 10:59 output): a ~25px webcam face was enlarged ~30x onto a 726px render face,
    producing a soft, blocky, translucent-looking face patch with smeared glasses inside the sharp render.
    A capture face that would need more than MAX_FACE_UPSCALE enlargement must never be composited.
    """

    REN = (1800, 2400)

    def setUp(self):
        self.render = textured(*self.REN, seed=9)
        self.ren_pts = ellipse_points(900, 700, 363, 450)  # ~726px face, like the real failing output

    def _restore(self, cap_face_px):
        cap = textured(360, 480, seed=8)
        r = cap_face_px / 2.0
        cap_pts = ellipse_points(180, 150, r, r * 1.24)
        pts = {(360, 480): cap_pts, self.REN: self.ren_pts}
        with patch.object(fp, "detect_face_points", side_effect=lambda rgb: pts.get((rgb.shape[1], rgb.shape[0]))), \
             patch.object(fp, "SEAM_MAX_RESIDUAL", 1.0), patch.object(fp, "MAX_FACE_TONE_DELTA_E", 1e9):
            return fp.restore_original_face(Image.fromarray(cap), _png(self.render))

    def test_tiny_capture_face_is_never_pasted(self):
        out, check = self._restore(25)
        self.assertEqual(out, _png(self.render), "no low-detail face patch may be composited")
        self.assertFalse(check.restored)
        self.assertTrue(check.low_detail_skip)
        self.assertGreater(check.details["upscale"], fp.MAX_FACE_UPSCALE)

    def test_limit_boundary(self):
        _, ok = self._restore(726 / (fp.MAX_FACE_UPSCALE - 0.2))
        _, skip = self._restore(726 / (fp.MAX_FACE_UPSCALE + 0.2))
        self.assertTrue(ok.restored)
        self.assertFalse(skip.restored)
        self.assertTrue(skip.low_detail_skip)

    def test_generation_serves_the_clean_render_instead_of_a_patch(self):
        """With a too-small capture face, generation returns the render untouched (one clean face), no retry."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from app.models import ClothAnalysis
        raw = _png(self.render)
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(
            candidates=[], prompt_feedback=None, sdk_http_response=None,
            parts=[SimpleNamespace(text=None, inline_data=SimpleNamespace(data=raw))])
        cloth = ClothAnalysis(category="shirt", garment_name="S", fabric_type="c", structure_level="soft_drape",
                              is_outerwear=False, sleeve_behavior="", tryon_guidelines="")
        skip = fp.FaceCheck(0.9, 0.5, False, "small", restored=False, low_detail_skip=True)
        with patch.object(gemini_client, "get_gemini_api_key", return_value="test-key-1234567890"), \
             patch.object(gemini_client.genai, "Client", return_value=client), \
             patch.object(gemini_client, "restore_original_face", return_value=(raw, skip)):
            out, _, _ = gemini_client.generate_tryon_image(user_img=Image.new("RGB", (60, 80)), dress_img=Image.new("RGB", (8, 8)),
                                                           cloth_analysis=cloth, user_appearance="a")
        self.assertEqual(out, raw)

    def test_diagnostic_log_line(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._restore(25)
            self._restore(300)
        lines = [l for l in buf.getvalue().splitlines() if l.startswith("FACE_RESTORE:")]
        self.assertEqual(len(lines), 2)
        for key in ("input_size=", "generated_size=", "face_bbox=", "upscale=", "face_restore_applied="):
            self.assertIn(key, lines[0])
        self.assertIn("face_restore_applied=False", lines[0])
        self.assertIn("face_restore_applied=True", lines[1])
        self.assertIn("transform_valid=True", lines[1])
        self.assertIn("working_resolution=1800x2400", lines[1])


class YoloNeverTouchesPixelsTest(unittest.TestCase):
    """Real YOLO detection leaves the image byte-for-byte unchanged and returns no image."""

    def test_detection_is_read_only(self):
        ASSETS = os.path.join(os.path.dirname(__file__), "assets")
        from app.services.yolo_service import YOLOService
        img = Image.open(os.path.join(ASSETS, "bus.jpg")).convert("RGB")
        arr = np.array(img)
        before_img, before_arr = img.tobytes(), arr.copy()
        det = YOLOService().detect_person(img)
        det_arr = YOLOService().detect_person(arr)
        self.assertEqual(img.tobytes(), before_img)
        self.assertTrue(np.array_equal(arr, before_arr))
        self.assertFalse(any(isinstance(v, (np.ndarray, Image.Image)) for v in det.to_dict().values()))
        self.assertFalse(any(isinstance(v, (np.ndarray, Image.Image)) for v in det_arr.to_dict().values()))


class RealFaceRestoreTest(unittest.TestCase):
    """Real MediaPipe landmarks on a real face (Ultralytics' bundled zidane.jpg)."""

    @classmethod
    def setUpClass(cls):
        ASSETS = os.path.join(os.path.dirname(__file__), "assets")
        full = np.array(Image.open(os.path.join(ASSETS, "zidane.jpg")).convert("RGB"))
        cls.capture = full[0:720, 740:1180].copy()  # portrait crop around the right-hand man
        h, w = cls.capture.shape[:2]
        # "Model render": same person on a 3x canvas, shifted, rotated 3 deg, re-lit (studio exposure)
        M = cv2.getRotationMatrix2D((w * 1.5, h * 1.5), 3.0, 1.0) @ np.vstack([[3, 0, 40], [0, 3, 60], [0, 0, 1]])
        big = cv2.warpAffine(cls.capture, M, (w * 3, h * 3), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cls.render = np.clip(big.astype(np.float32) * 1.12 + 10, 0, 255).astype(np.uint8)
        pts = fp.detect_face_points(cls.render)
        cls.align = fp.estimate_alignment(fp.detect_face_points(cls.capture), pts)
        # The model's own, different glasses drawn over the eyes: they must NOT survive restoration
        cls.fake_glasses = np.zeros(cls.render.shape[:2], np.uint8)
        for eye in ([33, 133], [362, 263]):
            (x1, y1), (x2, y2) = pts[eye[0]], pts[eye[1]]
            cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1) * 0.85
            cv2.rectangle(cls.fake_glasses, (int(cx - r), int(cy - r * 0.7)), (int(cx + r), int(cy + r * 0.7)), 255, 6)
        cls.render[cls.fake_glasses > 0] = (20, 20, 20)
        cls.out_bytes, cls.check = fp.restore_original_face(Image.fromarray(cls.capture), _png(cls.render))
        cls.out = _decode(cls.out_bytes)

    def test_restored_with_valid_alignment(self):
        self.assertTrue(self.check.restored, self.check.reason)
        self.assertAlmostEqual(self.check.details["scale"], 3.0, delta=0.15)
        self.assertLess(self.check.details["shape_rms_frac"], 0.03)
        self.assertEqual(self.out.shape[:2], self.render.shape[:2])

    def test_seam_agrees_for_the_same_person(self):
        self.assertLess(self.check.details["seam_residual"], fp.SEAM_MAX_RESIDUAL)

    def _truth(self):
        h, w = self.capture.shape[:2]
        return cv2.getRotationMatrix2D((w * 1.5, h * 1.5), 3.0, 1.0) @ np.vstack([[3, 0, 40], [0, 3, 60], [0, 0, 1]])

    def test_face_placed_where_it_truly_belongs(self):
        """Correct position, scale and rotation: landmark placement vs the known ground-truth transform."""
        pts_o = fp.detect_face_points(self.capture)
        face_w = fp._face_width(fp.detect_face_points(self.render))
        err = np.linalg.norm(fp._transform(pts_o, self.align.M) - fp._transform(pts_o, self._truth().astype(np.float32)), axis=1)
        self.assertLess(err.mean() / face_w, 0.015)
        self.assertLess(err.max() / face_w, 0.025)

    def test_inner_face_is_the_real_face(self):
        """
        Eyes/brows/nose/mouth pixels match the true (capture) face, relit by the reported colour adjustment. (Here the
        model's face is the same face relit plus its own glasses, so once the colour is matched the two differ only at
        the glasses: test_model_glasses_are_gone_no_duplicate_frames covers that.)
        """
        h, w = self.capture.shape[:2]
        truth = cv2.warpAffine(self.capture, self._truth(), (w * 3, h * 3), flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REPLICATE)
        truth = np.clip(fp._apply_tone(truth, fp.Tone(**self.check.details["tone"])), 0, 255).astype(np.uint8)
        pg = fp.detect_face_points(self.render)
        inner = np.zeros(self.out.shape[:2], np.uint8)
        cv2.fillPoly(inner, [cv2.convexHull(pg[fp.BROWS + fp.EYES + fp.NOSE + fp.LIPS].astype(np.int32))], 1)
        m = inner.astype(bool)
        mae_truth = np.abs(self.out[m].astype(int) - truth[m].astype(int)).mean()
        self.assertLess(mae_truth, 12.0)

    def test_independent_verifier_pixel_similarity(self):
        self.assertGreaterEqual(self.check.pixel_similarity, fp.FACE_PIXEL_SIM_THRESHOLD)

    def test_model_glasses_are_gone_no_duplicate_frames(self):
        """Where the model drew its own frames, the output must now show the real face, not frame pixels."""
        frames = self.fake_glasses > 0
        real = cv2.warpAffine(self.capture, self.align.M.astype(np.float64), self.render.shape[1::-1],
                              flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        out_l, real_l = self.out[frames].mean(1), real[frames].mean(1)
        survived = np.mean((out_l < 40) & (real_l > 80))  # dark in output although the real face is not dark there
        self.assertGreater(np.mean(self.render[frames].mean(1) < 40), 0.95)
        self.assertLess(survived, 0.01, f"{survived:.1%} of the model's frame pixels survived (duplicate glasses)")

    def test_exactly_one_face(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        path = next(p for p in (APP_DIR.parent / "assets" / "face_landmarker.task",) if p.exists())
        det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)), num_faces=4, min_face_detection_confidence=0.3))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(self.out)))
        self.assertEqual(len(res.face_landmarks), 1, "output must contain exactly one face")


class FaceColourMatchTest(unittest.TestCase):
    """
    Regression (30 Sep - 4 Oct test renders): the real face was pasted with the webcam's colour. Kiosk captures are far
    darker and greener than studio-lit renders, so the visitor got a dark olive face on a bright body, and the face
    check passed because it compares against the (equally dark) capture.
    """

    @classmethod
    def setUpClass(cls):
        ASSETS = os.path.join(os.path.dirname(__file__), "assets")
        full = np.array(Image.open(os.path.join(ASSETS, "zidane.jpg")).convert("RGB"))
        cls.studio = full[0:720, 740:1180].copy()
        h, w = cls.studio.shape[:2]
        M = cv2.getRotationMatrix2D((w * 1.5, h * 1.5), 3.0, 1.0) @ np.vstack([[3, 0, 40], [0, 3, 60], [0, 0, 1]])
        cls.render = cv2.warpAffine(cls.studio, M, (w * 3, h * 3), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # The same face as a dim, green-cast webcam shot (about the kiosk captures' darkness and cast)
        cls.webcam = np.clip(cls.studio * np.float32([0.72, 0.8, 0.7]), 0, 255).astype(np.uint8)
        pts = fp.detect_face_points(cls.render)
        x1, y1 = pts.min(0).astype(int)
        x2, y2 = pts.max(0).astype(int)
        fw, fh = x2 - x1, y2 - y1
        cls.cheeks = (slice(y1 + int(fh * 0.45), y1 + int(fh * 0.65)), slice(x1 + int(fw * 0.25), x1 + int(fw * 0.75)))

    def _cheek_delta_e(self, out):
        a, b = fp._lab(out[self.cheeks]).reshape(-1, 3).mean(0), fp._lab(self.render[self.cheeks]).reshape(-1, 3).mean(0)
        return float(np.linalg.norm(a - b))

    def test_dim_green_capture_gets_the_render_tone(self):
        out_bytes, check = fp.restore_original_face(Image.fromarray(self.webcam), _png(self.render))
        out = _decode(out_bytes)
        self.assertTrue(check.restored, check.reason)
        self.assertGreater(self._cheek_delta_e(self.webcam_on_render()), 2 * fp.MAX_FACE_TONE_DELTA_E,
                           "fixture must start visibly off-colour")
        self.assertLess(self._cheek_delta_e(out), 3.0, "pasted face must have the render's skin tone")
        self.assertLessEqual(check.details["tone_delta_e"], fp.MAX_FACE_TONE_DELTA_E)
        # Still the real face: the independent check passes against the capture with the same adjustment
        self.assertTrue(check.passed, check)

    def webcam_on_render(self):
        al = fp.estimate_alignment(fp.detect_face_points(self.webcam), fp.detect_face_points(self.render))
        return cv2.warpAffine(self.webcam, al.M.astype(np.float64), self.render.shape[1::-1], flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)

    def test_without_the_colour_match_the_gate_serves_the_clean_render(self):
        """The old behaviour (no colour match) must never reach a visitor: the off-colour paste is dropped."""
        with patch.object(fp, "_estimate_tone", return_value=fp.Tone()):
            out_bytes, check = fp.restore_original_face(Image.fromarray(self.webcam), _png(self.render))
        self.assertFalse(check.restored)
        self.assertTrue(check.low_detail_skip, "a clean render is safe to serve (no retry, no failure)")
        self.assertGreater(check.details["tone_delta_e"], fp.MAX_FACE_TONE_DELTA_E)
        self.assertIn("colour", check.reason)
        self.assertEqual(out_bytes, _png(self.render), "render must be returned untouched")

    def test_white_balance_correction_is_capped(self):
        """A cast bigger than the cap is not trusted as lighting; left half-corrected, it must not be pasted."""
        blue = np.clip(self.studio * np.float32([0.8, 0.85, 1.2]), 0, 255).astype(np.uint8)  # needs a b* shift of ~15
        with patch.object(fp, "TONE_MAX_CHROMA_SHIFT", 4.0):
            out_bytes, check = fp.restore_original_face(Image.fromarray(blue), _png(self.render))
        tone = check.details["tone"]
        self.assertLessEqual(abs(tone["a_shift"]), fp.TONE_MAX_CHROMA_SHIFT)
        self.assertLessEqual(abs(tone["b_shift"]), fp.TONE_MAX_CHROMA_SHIFT)
        self.assertFalse(check.restored, "a cast beyond the cap must not be pasted half-corrected")
        self.assertTrue(check.low_detail_skip)
        self.assertEqual(out_bytes, _png(self.render))

    def test_beauty_keeps_the_capture_tone_and_refuses_to_enlarge(self):
        """Save look (beauty): the real face is never relit (lightened) to the render; off-tone or enlarged = no paste."""
        _, check = fp.restore_original_face(Image.fromarray(self.studio), _png(self.render), keep_capture_tone=True)
        self.assertTrue(check.restored, check.reason)
        self.assertEqual(check.details["tone"], fp.Tone().as_details(), "the face keeps its own colour")
        _, check = fp.restore_original_face(Image.fromarray(self.webcam), _png(self.render), keep_capture_tone=True)
        self.assertFalse(check.restored, "a render brighter than the real skin must not be pasted onto it")
        _, check = fp.restore_original_face(Image.fromarray(self.studio), _png(self.render), keep_capture_tone=True,
                                            max_upscale=1.5)
        self.assertFalse(check.restored, "the fixture's face is enlarged ~3x")

    def test_same_lighting_needs_no_colour_change(self):
        _, check = fp.restore_original_face(Image.fromarray(self.studio), _png(self.render))
        tone = check.details["tone"]
        self.assertTrue(check.restored, check.reason)
        self.assertAlmostEqual(tone["l_scale"], 1.0, delta=0.08)
        self.assertLess(abs(tone["l_dst"] - tone["l_src"]), 2.0)
        self.assertLess(max(abs(tone["a_shift"]), abs(tone["b_shift"])), 1.5)


class FinalEncodeTest(unittest.TestCase):
    def test_png_encoded_once_to_rgb_jpeg_full_size(self):
        arr = textured(640, 480, 3)
        out, size = finalize_result_image(_png(arr))
        img = Image.open(io.BytesIO(out))
        self.assertEqual((img.format, img.mode, img.size, size), ("JPEG", "RGB", (640, 480), (640, 480)))

    def test_existing_rgb_jpeg_is_not_recompressed(self):
        buf = io.BytesIO()
        Image.fromarray(textured(64, 64, 4)).save(buf, format="JPEG", quality=95)
        out, _ = finalize_result_image(buf.getvalue())
        self.assertEqual(out, buf.getvalue())

    def test_alpha_is_removed(self):
        rgba = np.dstack([textured(64, 64, 5), np.full((64, 64), 128, np.uint8)])
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        out, _ = finalize_result_image(buf.getvalue())
        self.assertEqual(Image.open(io.BytesIO(out)).mode, "RGB")


class RouterFacePathTest(unittest.TestCase):
    """/api/tryon: the model's face is never served, and only the restored final image is cached."""

    W, H = 384, 512

    def setUp(self):
        from app.models import ClothAnalysis, SilhouetteVerificationResult
        img = np.full((self.H, self.W, 3), 238, np.uint8)
        cv2.rectangle(img, (130, 70), (250, 490), (90, 60, 40), -1)
        self.studio = img
        self.cloth = ClothAnalysis(category="shirt", garment_name="Shirt", fabric_type="cotton",
                                   structure_level="soft_drape", is_outerwear=False, sleeve_behavior="n",
                                   tryon_guidelines="n")
        self.passing = SilhouetteVerificationResult(score=0.9, passed=True, threshold=0.6, details={})

    def _uri(self, fmt="PNG"):
        buf = io.BytesIO()
        Image.fromarray(self.studio).save(buf, format=fmt)
        return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

    def _run(self, generate, has_face=True, session_id=None):
        import asyncio
        from fastapi import HTTPException
        from app.models import TryOnRequest
        from app.routers import tryon
        req = TryOnRequest(user_image=self._uri(), dress_image=self._uri(), cloth_analysis=self.cloth, session_id=session_id)
        with patch.object(tryon, "YOLO_VALIDATION_ENABLED", False), \
             patch.object(tryon, "_capture_has_face", return_value=has_face), \
             patch.object(tryon, "generate_tryon_image", side_effect=generate), \
             patch.object(tryon, "analyze_user_appearance", return_value=""), \
             patch.object(tryon, "prepare_user_portrait", side_effect=lambda im: im), \
             patch.object(tryon.silhouette_verifier, "verify", return_value=self.passing), \
             patch.object(tryon.body_profile_service, "extract_single_frame", return_value=None), \
             patch.object(tryon.body_profile_service, "extract_profile", side_effect=RuntimeError("n/a")), \
             patch.object(tryon.body_profile_service, "generate_face_and_hands_mask",
                          return_value=(None, np.zeros((self.H, self.W), np.float32))), \
             patch.object(tryon, "_face_score", return_value=0.99), \
             patch.object(tryon, "apply_watermark"), \
             patch.object(tryon, "generate_qr_data_uri", return_value="qr"):
            try:
                return asyncio.run(tryon.virtual_tryon(req))
            except HTTPException as exc:
                return exc

    def _restored_bmp(self):
        buf = io.BytesIO()
        Image.fromarray(self.studio).save(buf, format="BMP")
        return buf.getvalue()

    def test_no_face_in_capture_rejected_before_gemini(self):
        from fastapi import HTTPException
        from app.routers import tryon
        calls = []
        resp = self._run(lambda **kw: calls.append(kw), has_face=False)
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual((resp.status_code, resp.detail), (422, tryon.NO_FACE_MESSAGE))
        self.assertEqual(calls, [])

    def test_unrestorable_render_is_regenerated_not_served(self):
        from fastapi import HTTPException
        calls = []

        def generate(**kw):
            calls.append(kw)
            if len(calls) == 1:
                raise HTTPException(status_code=502, detail=gemini_client.FACE_NOT_RESTORED_MESSAGE)
            return self._restored_bmp(), 0.97, self.cloth

        resp = self._run(generate)
        self.assertTrue(resp.success)
        self.assertEqual(len(calls), 2)

    def test_never_serves_when_no_attempt_could_be_restored(self):
        from fastapi import HTTPException

        def generate(**kw):
            raise HTTPException(status_code=502, detail=gemini_client.FACE_NOT_RESTORED_MESSAGE)

        resp = self._run(generate)
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual(resp.detail, gemini_client.FACE_NOT_RESTORED_MESSAGE)

    def test_lossless_render_encoded_once_to_full_size_rgb_jpeg(self):
        resp = self._run(lambda **kw: (self._restored_bmp(), 0.97, self.cloth))
        img = Image.open(io.BytesIO(base64.b64decode(resp.result_image.split(",", 1)[1])))
        self.assertEqual((img.format, img.mode, img.size), ("JPEG", "RGB", (self.W, self.H)))

    def test_cached_canonical_is_exactly_the_served_restored_image(self):
        from app.services.session_manager import session_manager
        resp = self._run(lambda **kw: (self._restored_bmp(), 0.97, self.cloth), session_id="face-cache-test")
        session = session_manager.get_session("face-cache-test")
        self.assertEqual(session.get_canonical_render(dress_hash=gemini_client.get_dress_hash(Image.fromarray(self.studio))[:16]),
                         resp.result_image)
        session_manager.purge_session("face-cache-test")

    def test_no_cache_written_when_generation_fails(self):
        from fastapi import HTTPException
        from app.services.session_manager import session_manager

        def generate(**kw):
            raise HTTPException(status_code=502, detail=gemini_client.FACE_NOT_RESTORED_MESSAGE)

        self._run(generate, session_id="face-nocache-test")
        session = session_manager.get_session("face-nocache-test")
        dress_hash = gemini_client.get_dress_hash(Image.fromarray(self.studio))[:16]
        self.assertIsNone(session.get_canonical_render(dress_hash=dress_hash))
        self.assertIsNone(session.get_last_good_render())
        session_manager.purge_session("face-nocache-test")


class CaptureTimeFaceCheckTest(unittest.TestCase):
    """/api/check-face: the pre-generation face check, run when the photo is taken so the visitor retakes it there."""

    def _uri(self, arr):
        return "data:image/png;base64," + base64.b64encode(_png(arr)).decode()

    def _run(self, image_uri, **patches):
        import asyncio
        from contextlib import nullcontext
        from fastapi import HTTPException
        from app.models import FaceCheckRequest
        from app.routers import tryon
        with patch.multiple(tryon, **patches) if patches else nullcontext():
            try:
                return asyncio.run(tryon.check_face(FaceCheckRequest(image=image_uri)))
            except HTTPException as exc:
                return exc

    def test_hidden_face_gets_the_generation_step_message(self):
        from fastapi import HTTPException
        from app.routers import tryon
        blank = np.full((512, 384, 3), 238, np.uint8)  # real MediaPipe: nothing to detect
        resp = self._run(self._uri(blank))
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual((resp.status_code, resp.detail), (422, tryon.NO_FACE_MESSAGE))

    def test_visible_face_passes(self):
        resp = self._run(self._uri(np.zeros((512, 384, 3), np.uint8)), _capture_has_face=lambda im: True)
        self.assertEqual(resp, {"success": True})

    def test_checks_the_same_portrait_crop_as_generation(self):
        crop = Image.new("RGB", (30, 40))
        seen = []
        self._run(self._uri(np.zeros((360, 640, 3), np.uint8)),
                  prepare_user_portrait=lambda im: crop, _capture_has_face=lambda im: seen.append(im) or True)
        self.assertEqual(seen, [crop])

    def test_unreadable_image(self):
        from fastapi import HTTPException
        resp = self._run("data:image/png;base64,bm90LWFuLWltYWdl")
        self.assertIsInstance(resp, HTTPException)
        self.assertEqual(resp.status_code, 400)


class SingleFacePathStaticGuard(unittest.TestCase):
    """ONE face-compositing path: restore_original_face, called once, from gemini_client only."""

    def _src(self, rel):
        return (APP_DIR / rel).read_text(encoding="utf-8")

    def test_no_translucent_blend_apis_anywhere(self):
        pattern = re.compile(r"\b(Image\.blend|addWeighted|seamlessClone|alpha_composite|Image\.composite)\s*\(")
        for path in APP_DIR.rglob("*.py"):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), f"blend API in {path.name}")

    def test_face_compositing_only_in_face_preservation(self):
        for path in APP_DIR.rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            if path.name != "face_preservation.py":
                self.assertNotIn("warpAffine(", src, f"warpAffine in {path.name}")
                self.assertNotIn("_composite(", src, f"face compositing in {path.name}")

    def test_restore_called_exactly_once_and_only_by_generation(self):
        callers = sorted(p.name for p in APP_DIR.rglob("*.py")
                         if "restore_original_face(" in p.read_text(encoding="utf-8") and p.name != "face_preservation.py")
        # beauty.py: the saved beauty look's studio portrait. Same rule: one paste, onto the model's raw output.
        self.assertEqual(callers, ["beauty.py", "gemini_client.py"])
        beauty_calls = [n for n in ast.walk(ast.parse(self._src("routers/beauty.py")))
                        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "restore_original_face"]
        self.assertEqual(len(beauty_calls), 1)
        self.assertEqual(getattr(beauty_calls[0].args[1], "id", None), "render")
        tree = ast.parse(self._src("services/gemini_client.py"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "restore_original_face"]
        # The capture restore, plus the face lock that runs only when the capture face was too small to composite.
        # Both composite onto the model's raw output, never onto an already-composited image: at most one face paste.
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(getattr(call.args[1], "id", None), "generated_image_bytes")

    def test_unrestored_render_is_never_returned_by_generation(self):
        """generate_tryon_image raises instead of returning a render whose face could not be restored."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        from app.models import ClothAnalysis
        raw = _png(textured(60, 80, 7))
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(
            candidates=[], prompt_feedback=None, sdk_http_response=None,
            parts=[SimpleNamespace(text=None, inline_data=SimpleNamespace(data=raw))])
        cloth = ClothAnalysis(category="shirt", garment_name="S", fabric_type="c", structure_level="soft_drape",
                              is_outerwear=False, sleeve_behavior="", tryon_guidelines="")
        with patch.object(gemini_client, "get_gemini_api_key", return_value="test-key-1234567890"), \
             patch.object(gemini_client.genai, "Client", return_value=client), \
             patch.object(gemini_client, "restore_original_face",
                          return_value=(raw, fp.FaceCheck(0.5, 0.4, False, "stub", restored=False))):
            with self.assertRaises(HTTPException) as ctx:
                gemini_client.generate_tryon_image(user_img=Image.new("RGB", (60, 80)), dress_img=Image.new("RGB", (8, 8)),
                                                   cloth_analysis=cloth, user_appearance="a")
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.detail, gemini_client.FACE_NOT_RESTORED_MESSAGE)

    def test_capture_face_restored_not_reference_render(self):
        tree = ast.parse(self._src("services/gemini_client.py"))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "restore_original_face")
        self.assertEqual(call.args[0].id, "user_portrait")


if __name__ == "__main__":
    unittest.main()
