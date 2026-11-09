import base64
import io
import unittest

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.models import LipstickFinish
from app.routers import beauty
from app.services import beauty_studio_service as studio
from app.services import lipstick_service as ls

W, H = 400, 400
CX, CY = 200.0, 260.0


def _loop(indices, rx, ry_top, ry_bottom, cx=CX, cy=CY):
    """Places a closed lip contour (MediaPipe order: corner, upper lip to the other corner, lower lip back)."""
    n = len(indices)
    half = n // 2
    pts = {}
    for k, idx in enumerate(indices):
        if k <= half:  # upper lip, left corner -> right corner
            t = np.pi - np.pi * k / half
            pts[idx] = (cx + rx * np.cos(t), cy - ry_top * np.sin(t))
        else:  # lower lip, right corner -> left corner
            t = -np.pi * (k - half) / (n - half)
            pts[idx] = (cx + rx * np.cos(t), cy - ry_bottom * np.sin(t))
    return pts


def _face_points(mouth_open=True, rx=60.0):
    """A synthetic 478-point mesh: face oval on a circle, real lip contours, everything else at the face centre."""
    pts = np.tile(np.array([[CX, CY - 60.0]], dtype=np.float32), (478, 1))
    for k, idx in enumerate(ls.FACE_OVAL):
        t = 2 * np.pi * k / len(ls.FACE_OVAL)
        pts[idx] = (CX + 150 * np.sin(t), CY - 60 - 170 * np.cos(t))
    for idx, p in _loop(ls.LIPS_OUTER, rx, rx * 0.40, rx * 0.45).items():
        pts[idx] = p
    inner_ry = (rx * 0.18, rx * 0.20) if mouth_open else (0.5, 0.5)
    for idx, p in _loop(ls.LIPS_INNER, rx * 0.80, *inner_ry).items():
        pts[idx] = p
    return pts


def _portrait(seed=0):
    """Skin-toned image with pinkish, textured lips and white 'teeth' in the mouth opening."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (196, 150, 120), dtype=np.float32)
    pts = _face_points()
    outer = np.zeros((H, W), np.uint8)
    cv2.fillPoly(outer, [np.round(pts[ls.LIPS_OUTER]).astype(np.int32)], 1)
    inner = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inner, [np.round(pts[ls.LIPS_INNER]).astype(np.int32)], 1)
    img[outer > 0] = (185, 110, 110)
    # Vertical lip creases: a lightness texture that must survive recolouring
    creases = 18 * np.sin(np.arange(W) / 3.0)[None, :, None]
    img = np.where(outer[..., None] > 0, img + creases, img)
    img[inner > 0] = (245, 245, 240)
    img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _masks(pts):
    outer = np.zeros((H, W), np.uint8)
    cv2.fillPoly(outer, [np.round(pts[ls.LIPS_OUTER]).astype(np.int32)], 1)
    inner = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inner, [np.round(pts[ls.LIPS_INNER]).astype(np.int32)], 1)
    # Interior of the lip band only (away from both edges), where the shade is fully applied
    body = cv2.erode(outer, np.ones((7, 7), np.uint8)) & ~cv2.dilate(inner, np.ones((7, 7), np.uint8)) & 1
    inner_core = cv2.erode(inner, np.ones((3, 3), np.uint8))
    return body.astype(bool), inner_core.astype(bool), outer.astype(bool)


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


class LipMaskTest(unittest.TestCase):
    def test_mouth_opening_is_excluded_and_edges_are_feathered(self):
        pts = _face_points(mouth_open=True)
        mask = ls.build_lip_mask((H, W), pts)
        body, inner_core, outer = _masks(pts)
        self.assertTrue(np.all(mask[inner_core] == 0.0), "teeth/tongue must get no colour at all")
        self.assertGreater(mask[body].mean(), 0.95)
        partial = (mask > 0.05) & (mask < 0.95)
        self.assertGreater(partial.sum(), 0, "outer edge should be soft, not a hard cut-out")
        far = np.ones((H, W), bool)
        far[150:350, 100:300] = False
        self.assertEqual(mask[far].max(), 0.0)

    def test_closed_mouth_tints_the_whole_lip_including_the_seam(self):
        pts = _face_points(mouth_open=False)
        mask = ls.build_lip_mask((H, W), pts)
        _, _, outer = _masks(pts)
        self.assertGreater(mask[outer].mean(), 0.75)
        seam = np.round(pts[[13, 14]].mean(axis=0)).astype(int)
        self.assertGreater(mask[seam[1], seam[0]], 0.95, "no unpainted line where the lips meet")

    def test_lip_line_is_smooth(self):
        curve = ls.smooth_contour(_face_points()[ls.LIPS_OUTER])
        self.assertEqual(len(curve), len(ls.LIPS_OUTER) * ls.SPLINE_SAMPLES)
        # The spline passes through every landmark
        pts = _face_points()[ls.LIPS_OUTER]
        np.testing.assert_allclose(curve[:: ls.SPLINE_SAMPLES], pts, atol=1e-3)


class ApplyLipstickTest(unittest.TestCase):
    def setUp(self):
        self.rgb = _portrait()
        self.pts = _face_points()
        self.body, self.inner_core, self.outer = _masks(self.pts)

    def test_teeth_and_skin_outside_the_lips_are_untouched(self):
        out = ls.apply_lipstick(self.rgb, self.pts, "#9E1A2A", LipstickFinish.MATTE, 1.0)
        self.assertEqual(out.shape, self.rgb.shape)
        np.testing.assert_array_equal(out[self.inner_core], self.rgb[self.inner_core])
        far = np.ones((H, W), bool)
        far[150:350, 60:340] = False
        np.testing.assert_array_equal(out[far], self.rgb[far])

    def test_lip_colour_moves_to_the_shade(self):
        target = ls.hex_to_rgb("#581845")
        out = ls.apply_lipstick(self.rgb, self.pts, "#581845", LipstickFinish.SATIN, 1.0)
        t_lab = _lab(target.reshape(1, 1, 3))[0, 0]
        before = np.linalg.norm(_lab(self.rgb)[self.body][:, 1:] - t_lab[1:], axis=1).mean()
        after = np.linalg.norm(_lab(out)[self.body][:, 1:] - t_lab[1:], axis=1).mean()
        # Full = 85% coverage with a little of the lip's own tone showing through (beauty_tuning), not an opaque paint
        self.assertLess(after, before * 0.5)
        self.assertLess(abs(_lab(out)[self.body][:, 0].mean() - t_lab[0]), 15)

    def test_lip_texture_survives(self):
        out = ls.apply_lipstick(self.rgb, self.pts, "#C1121F", LipstickFinish.SATIN, 1.0)
        # Fine detail only: the finish adds broad sheen on purpose, the creases must stay where they were
        def fine(rgb):
            L = _lab(rgb)[..., 0]
            return (L - cv2.GaussianBlur(L, (0, 0), 3))[self.body]
        # (a flat "sticker" scores ~0; cream softens tiny highlights on purpose, so not ~1 either)
        self.assertGreater(np.corrcoef(fine(self.rgb), fine(out))[0, 1], 0.85, "creases must stay where they were")
        L_in = _lab(self.rgb)[..., 0][self.body]
        L_out = _lab(out)[..., 0][self.body]
        self.assertGreater(L_out.std(), L_in.std() * 0.6, "not a flat sticker")

    def test_intensity_scales_the_effect(self):
        none = ls.apply_lipstick(self.rgb, self.pts, "#9E1A2A", LipstickFinish.SATIN, 0.0)
        np.testing.assert_array_equal(none, self.rgb)
        diffs = []
        for intensity in (0.3, 0.6, 1.0):
            out = ls.apply_lipstick(self.rgb, self.pts, "#9E1A2A", LipstickFinish.SATIN, intensity)
            diffs.append(np.abs(out[self.body].astype(int) - self.rgb[self.body]).mean())
        self.assertLess(diffs[0], diffs[1])
        self.assertLess(diffs[1], diffs[2])

    def test_lowest_slider_setting_is_a_sheer_tint(self):
        from app.services.beauty_tuning import LIP_OPACITY_MAX, LIP_OPACITY_MIN, opacity
        self.assertEqual(opacity(0.0, LIP_OPACITY_MIN, LIP_OPACITY_MAX), 0.0)
        self.assertAlmostEqual(opacity(0.01, LIP_OPACITY_MIN, LIP_OPACITY_MAX), LIP_OPACITY_MIN, places=2)
        self.assertAlmostEqual(opacity(1.0, LIP_OPACITY_MIN, LIP_OPACITY_MAX), LIP_OPACITY_MAX)

        def delta_e(t):
            out = ls.apply_lipstick(self.rgb, self.pts, "#5A1A28", LipstickFinish.MATTE, t)
            return np.linalg.norm(_lab(out)[self.body] - _lab(self.rgb)[self.body], axis=1).mean()
        # Lowest = slider 1: the lip's own colour must stay clearly visible (a tinted balm), far from the default
        self.assertLess(delta_e(0.01), delta_e(0.75) * 0.2)

    def test_light_shade_on_pigmented_lips_shows_the_lip_tone(self):
        dark = self.rgb.copy()
        dark[self.outer] = (dark[self.outer] * 0.45).astype(np.uint8)
        out = ls.apply_lipstick(dark, self.pts, "#B5756B", LipstickFinish.MATTE, 1.0)
        L_shade = _lab(ls.hex_to_rgb("#B5756B").reshape(1, 1, 3))[0, 0, 0]
        L_out = _lab(out)[..., 0][self.body].mean()
        self.assertLess(L_out, L_shade - 5, "a nude on dark lips reads deeper than its swatch, not like paint")
        self.assertGreater(L_out, _lab(dark)[..., 0][self.body].mean(), "but it still lightens them")

    def test_finishes_shape_the_highlights(self):
        top = {}
        for finish in LipstickFinish:
            out = ls.apply_lipstick(self.rgb, self.pts, "#C46D75", finish, 1.0)
            top[finish] = np.percentile(_lab(out)[..., 0][self.body], 99)
        self.assertLess(top[LipstickFinish.MATTE], top[LipstickFinish.SATIN])
        self.assertLess(top[LipstickFinish.SATIN] + 5, top[LipstickFinish.GLOSS], "gloss needs a clearly brighter shine")

    def test_finishes_are_visibly_different(self):
        outs = {f: _lab(ls.apply_lipstick(self.rgb, self.pts, "#9E1A2A", f, 1.0))[self.body] for f in LipstickFinish}
        pairs = [(LipstickFinish.MATTE, LipstickFinish.SATIN), (LipstickFinish.SATIN, LipstickFinish.GLOSS),
                 (LipstickFinish.MATTE, LipstickFinish.GLOSS)]
        for a, b in pairs:
            delta_e = np.linalg.norm(outs[a] - outs[b], axis=1)
            self.assertGreater(np.percentile(delta_e, 95), 4.0, f"{a.value} vs {b.value} look the same")

    def test_matte_has_no_shine(self):
        out = ls.apply_lipstick(self.rgb, self.pts, "#C46D75", LipstickFinish.MATTE, 1.0)
        L = _lab(out)[..., 0][self.body]
        self.assertLess(np.percentile(L, 99) - np.median(L), 8.0)

    def test_only_three_finishes(self):
        self.assertEqual({f.value for f in LipstickFinish}, {"matte", "satin", "gloss"})

    def test_input_is_not_modified(self):
        before = self.rgb.copy()
        ls.apply_lipstick(self.rgb, self.pts, "#9E1A2A", LipstickFinish.GLOSS, 1.0)
        np.testing.assert_array_equal(self.rgb, before)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            ls.apply_lipstick(self.rgb, self.pts[:100], "#9E1A2A")
        with self.assertRaises(ValueError):
            ls.hex_to_rgb("#12345")


class TryOnLipstickTest(unittest.TestCase):
    def _run(self, pts):
        original = ls.detect_lip_points
        ls.detect_lip_points = lambda rgb: pts
        try:
            return ls.try_on_lipstick(Image.fromarray(_portrait()), "#9E1A2A", LipstickFinish.MATTE, 0.8)
        finally:
            ls.detect_lip_points = original

    def test_no_face(self):
        with self.assertRaises(ls.LipsNotFound):
            self._run(None)

    def test_face_too_small_for_a_lip_preview(self):
        with self.assertRaises(ls.LipsNotFound) as ctx:
            self._run(_face_points(rx=6.0))
        self.assertIn("close-up", str(ctx.exception))

    def test_returns_image_and_normalized_face_box(self):
        res = self._run(_face_points())
        self.assertEqual(res.image.shape, (H, W, 3))
        self.assertAlmostEqual(res.mouth_width_px, 120.0, delta=1.0)
        for key in ("x", "y", "w", "h"):
            self.assertGreaterEqual(res.face_box[key], 0.0)
            self.assertLessEqual(res.face_box[key], 1.0)


def _b64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


LIP = {"shade_id": "ruby_royale", "shade_name": "Ruby Royale", "hex_color": "#9E1A2A", "finish": "matte",
       "intensity": 0.85}
POWDER = {"shade_id": "honey", "shade_name": "Honey", "hex_tint": "#C08A5E", "finish": "natural_velvet",
          "coverage": "medium"}


class BeautyRouterTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(beauty.router)
        self.client = TestClient(app)
        self._detect, self._save = ls.detect_lip_points, beauty._save_download
        self._available = studio.studio_render_available
        ls.detect_lip_points = lambda rgb: _face_points()
        self.saved = []
        beauty._save_download = lambda rgb, _session=None: self.saved.append(rgb) or "data:image/png;base64,QR"
        studio.studio_render_available = lambda: False  # never a real (paid) Gemini call from the tests

    def tearDown(self):
        ls.detect_lip_points, beauty._save_download = self._detect, self._save
        studio.studio_render_available = self._available

    def _post(self, **body):
        return self.client.post("/api/beauty/tryon", json={"image": _b64(_portrait()), **body})

    def test_shades_catalog(self):
        shades = self.client.get("/api/beauty/shades").json()
        self.assertEqual(len(shades), 11)
        self.assertEqual(len({s["id"] for s in shades}), 11)
        self.assertEqual({s["category"] for s in shades}, {"Roses", "Reds", "Berries & Wines", "Browns"})
        self.assertEqual({s["finish_default"] for s in shades}, {"matte", "satin", "gloss"})

    def test_powder_catalog(self):
        powders = self.client.get("/api/beauty/powders").json()
        self.assertEqual(len(powders), 8)
        self.assertEqual(powders[0]["id"], "warm_ivory")
        self.assertTrue(all(p["tinted"] for p in powders))
        self.assertTrue(all(p["hex_tint"].startswith("#") for p in powders))

    def test_three_modes(self):
        for body, mode in (({"lipstick": LIP}, "lipstick_only"), ({"compact_powder": POWDER}, "powder_only"),
                           ({"lipstick": LIP, "compact_powder": POWDER}, "full_makeover")):
            res = self._post(**body).json()
            self.assertTrue(res["success"], res)
            self.assertEqual(res["mode"], mode)
            self.assertTrue(res["result_image"].startswith("data:image/jpeg;base64,"))
            self.assertEqual(res["active_look"]["lipstick"] is not None, "lipstick" in body)
            self.assertEqual(res["active_look"]["compact_powder"] is not None, "compact_powder" in body)
            self.assertIsNone(res["qr_code_base64"])
        self.assertEqual(self.saved, [], "previews write no download copy")

    def test_nothing_selected_is_rejected(self):
        self.assertEqual(self._post().status_code, 422)

    def test_save_look_returns_a_qr(self):
        res = self._post(lipstick=LIP, compact_powder=POWDER, include_qr=True).json()
        self.assertTrue(res["success"], res)
        self.assertEqual(res["qr_code_base64"], "data:image/png;base64,QR")
        self.assertEqual(len(self.saved), 1)
        self.assertFalse(res["studio_render"])

    def test_no_face_is_a_message_not_an_error(self):
        ls.detect_lip_points = lambda rgb: None
        res = self._post(compact_powder=POWDER).json()
        self.assertFalse(res["success"])
        self.assertEqual(res["mode"], "powder_only")
        self.assertEqual(res["result_image"], "")
        self.assertTrue(res["message"])

    def test_validation(self):
        bad_cases = [
            {"lipstick": {**LIP, "hex_color": "red"}},
            {"lipstick": {**LIP, "intensity": 1.5}},
            {"lipstick": {**LIP, "finish": "shimmer"}},
            {"compact_powder": {**POWDER, "finish": "dewy"}},
            {"compact_powder": {**POWDER, "coverage": "full"}},
        ]
        for bad in bad_cases:
            self.assertEqual(self._post(**bad).status_code, 422, bad)
        res = self.client.post("/api/beauty/tryon", json={"image": "not-an-image", "lipstick": LIP}).json()
        self.assertFalse(res["success"])


class BeautyServiceTest(unittest.TestCase):
    def setUp(self):
        from app.models import CompactPowderConfig, LipstickConfig
        from app.services import beauty_service
        self.bs = beauty_service
        self.lip, self.powder = LipstickConfig(**LIP), CompactPowderConfig(**POWDER)
        self.rgb, self.pts = _portrait(), _face_points()
        self.body, self.inner_core, self.outer = _masks(self.pts)

    def test_lipstick_only_leaves_skin_alone(self):
        out = self.bs.apply_look(self.rgb, self.pts, self.lip, None)
        skin = np.zeros((H, W), bool)
        skin[60:120, 120:280] = True
        np.testing.assert_array_equal(out[skin], self.rgb[skin])
        self.assertGreater(np.abs(out[self.body].astype(int) - self.rgb[self.body]).mean(), 10)

    def test_powder_only_leaves_lips_alone(self):
        out = self.bs.apply_look(self.rgb, self.pts, None, self.powder)
        self.assertLess(np.abs(out[self.body].astype(int) - self.rgb[self.body]).mean(), 1.0)

    def test_full_makeover_is_powder_then_lipstick(self):
        both = self.bs.apply_look(self.rgb, self.pts, self.lip, self.powder)
        lip_only = self.bs.apply_look(self.rgb, self.pts, self.lip, None)
        # Lips carry the lipstick either way; the skin is where the powder shows
        self.assertLess(np.abs(both[self.body].astype(int) - lip_only[self.body]).mean(), 3.0)

    def test_coverage_and_translucent(self):
        from app.models import CompactPowderConfig
        sheer = self.bs.apply_look(self.rgb, self.pts, None, self.powder.model_copy(update={"coverage": "sheer"}))
        medium = self.bs.apply_look(self.rgb, self.pts, None, self.powder)
        cheek = (slice(200, 230), slice(100, 140))
        d_sheer = np.abs(sheer[cheek].astype(int) - self.rgb[cheek]).mean()
        d_medium = np.abs(medium[cheek].astype(int) - self.rgb[cheek]).mean()
        self.assertLessEqual(d_sheer, d_medium)
        translucent = CompactPowderConfig(shade_id="translucent", shade_name="Translucent", hex_tint="#F3EEE8")
        self.assertIsNone(self.bs.powder_tint(translucent))
        self.assertEqual(self.bs.powder_tint(self.powder), "#C08A5E")

    def test_mode_names(self):
        from app.models import BeautyTryOnRequest
        self.assertEqual(self.bs.look_mode(BeautyTryOnRequest(image="x", lipstick=self.lip)), "lipstick_only")
        self.assertEqual(self.bs.look_mode(BeautyTryOnRequest(image="x", compact_powder=self.powder)), "powder_only")
        self.assertEqual(self.bs.look_mode(BeautyTryOnRequest(image="x", lipstick=self.lip,
                                                              compact_powder=self.powder)), "full_makeover")


class PowderTest(unittest.TestCase):
    """Synthetic face: even skin, a dark 'beard' patch on the chin, dark eyes/brows, plus a shiny forehead spot."""

    def setUp(self):
        from app.services import powder_service as ps
        self.ps = ps
        self.pts = _face_points()
        img = np.full((H, W, 3), (200, 150, 120), dtype=np.float32)
        rng = np.random.default_rng(1)
        img += rng.normal(0, 3, img.shape)  # pores
        self.beard = (slice(318, 345), slice(165, 235))
        img[self.beard] = (70, 50, 40)
        self.shine = (slice(110, 130), slice(185, 215))
        img[self.shine] = (245, 215, 195)
        for idx in ps.LEFT_EYE_ZONE + ps.RIGHT_EYE_ZONE:  # place eyes/brows and paint them dark
            side = -1 if idx in ps.LEFT_EYE_ZONE else 1
            self.pts[idx] = (CX + side * 55 + (idx % 7) * 2, 160 + (idx % 5) * 3)
        self.eye = (slice(162, 170), slice(CX_INT - 55 - 1, CX_INT - 55 + 10))
        img[self.eye] = (40, 30, 30)
        for idx, p in zip(ps.SKIN_SAMPLES,
                          [(140, 230), (260, 230), (200, 150)]):
            self.pts[idx] = p
        self.pts[234], self.pts[454] = (50, 200), (350, 200)
        self.rgb = np.clip(img, 0, 255).astype(np.uint8)

    def test_beard_eyes_and_lips_get_no_powder(self):
        out = self.ps.apply_powder(self.rgb, self.pts, "#F1D6C3", 1.0)
        for region in (self.beard, self.eye):
            diff = np.abs(out[region].astype(int) - self.rgb[region]).mean()
            self.assertLess(diff, 1.5, "no powder on facial hair or eyes")
        _, _, lips = _masks(self.pts)
        self.assertLess(np.abs(out[lips].astype(int) - self.rgb[lips]).mean(), 1.0)

    def test_shine_is_controlled_and_skin_is_powdered(self):
        out = self.ps.apply_powder(self.rgb, self.pts, None, 1.0)
        L_in, L_out = _lab(self.rgb)[..., 0], _lab(out)[..., 0]
        # Powder is capped at 50% opacity (beauty_tuning.POWDER_OPACITY_MAX): a light layer, still visibly mattifying
        self.assertLess(L_out[self.shine].mean(), L_in[self.shine].mean() - 2, "forehead glare should be mattified")
        cheek = (slice(220, 240), slice(130, 150))
        # The cheek is powdered skin (translucent on already even, matte skin rightly changes nothing visible)
        self.assertGreater(self.ps.build_skin_mask(_lab(self.rgb), self.pts)[cheek].mean(), 0.8)
        # Translucent adds no colour: chroma stays where it was on plain skin
        self.assertLess(np.abs(_lab(out)[cheek][..., 1:] - _lab(self.rgb)[cheek][..., 1:]).mean(), 2.0)

    def test_tinted_powder_moves_skin_toward_the_shade(self):
        out = self.ps.apply_powder(self.rgb, self.pts, "#7A4E34", 1.0)
        cheek = (slice(220, 240), slice(130, 150))
        t = _lab(ls.hex_to_rgb("#7A4E34").reshape(1, 1, 3))[0, 0]
        before = np.abs(_lab(self.rgb)[cheek][..., 0] - t[0]).mean()
        after = np.abs(_lab(out)[cheek][..., 0] - t[0]).mean()
        self.assertLess(after, before)

    def test_outside_the_face_is_untouched(self):
        out = self.ps.apply_powder(self.rgb, self.pts, "#D6A985", 1.0)
        np.testing.assert_array_equal(out[:40], self.rgb[:40])
        np.testing.assert_array_equal(out[:, :40], self.rgb[:, :40])

    def test_finishes_are_distinct(self):
        from app.models import CompactPowderFinish as F
        out = {f: _lab(self.ps.apply_powder(self.rgb, self.pts, None, 1.0, f))[..., 0] for f in F}
        shine = {f: out[f][self.shine].mean() for f in F}
        self.assertLess(shine[F.MATTE], shine[F.NATURAL_VELVET], "matte removes the most shine")
        self.assertLess(shine[F.NATURAL_VELVET], shine[F.LUMINOUS], "luminous keeps a glow")
        cheek = (slice(220, 240), slice(130, 150))
        self.assertGreater(out[F.LUMINOUS][cheek].mean(), out[F.MATTE][cheek].mean())


CX_INT = int(CX)


class StudioPromptTest(unittest.TestCase):
    def setUp(self):
        from app.models import CompactPowderConfig, LipstickConfig
        self.lip, self.powder = LipstickConfig(**LIP), CompactPowderConfig(**POWDER)

    def test_full_makeover_prompt(self):
        p = studio.build_prompt(self.lip, self.powder)
        self.assertIn('Shade "Ruby Royale", target color #9E1A2A, Matte finish', p)
        self.assertIn("Full 85% rich pigment", p)
        self.assertIn('Compact powder: "Honey" (shade color #C08A5E), Natural Velvet finish', p)
        self.assertIn("medium, buildable coverage", p)
        self.assertIn("PRIORITY #2 — COMPACT POWDER", p)
        self.assertIn("PRIORITY #3 — LIPSTICK", p)
        self.assertIn("PRIORITY #5 — MASTER", p)
        self.assertNotIn("{", p)
        self.assertNotIn("Shimmer", p)

    def test_lipstick_only_prompt(self):
        p = studio.build_prompt(self.lip.model_copy(update={"finish": "gloss", "intensity": 0.5}), None)
        self.assertIn("High-Shine Gloss", p)
        self.assertIn("Sheer 50% tint", p)
        self.assertIn("Compact powder: none", p)
        self.assertNotIn("BEARD & FACIAL HAIR LOCK", p)
        self.assertIn("PRIORITY #2 — LIPSTICK", p)
        self.assertIn("PRIORITY #4 — MASTER", p)

    def test_powder_only_prompt(self):
        p = studio.build_prompt(None, self.powder.model_copy(update={"finish": "luminous", "coverage": "sheer"}), False)
        self.assertIn("Lipstick: none", p)
        self.assertNotIn("LIPSTICK APPLICATION RULES", p)
        self.assertIn("colorless, adds no tint", p)
        self.assertIn("Luminous finish", p)
        self.assertIn("sheer, barely-there coverage", p)
        self.assertIn("BEARD & FACIAL HAIR LOCK", p)

    def test_portrait_crop_is_3_by_4_around_the_face(self):
        img = Image.new("RGB", (1080, 1920))
        crop = studio.portrait_crop(img, {"x": 0.45, "y": 0.10, "w": 0.10, "h": 0.07})
        self.assertAlmostEqual(crop.width / crop.height, 0.75, delta=0.01)
        self.assertLess(crop.width, 400)


class SaveLookStudioTest(unittest.TestCase):
    """Save look with Gemini and face restore mocked: studio success and every fallback path."""

    def setUp(self):
        app = FastAPI()
        app.include_router(beauty.router)
        self.client = TestClient(app)
        self.orig = (ls.detect_lip_points, beauty._save_download, studio.studio_render_available,
                     studio.render_studio_portrait, beauty.restore_original_face)
        ls.detect_lip_points = lambda rgb: _face_points() * (rgb.shape[1] / W)
        beauty._save_download = lambda rgb, _session=None: "data:image/png;base64,QR"
        studio.studio_render_available = lambda: True
        self.calls = []

        def fake_render(portrait, prompt, session_id=None):
            self.calls.append(prompt)
            buf = io.BytesIO()
            portrait.save(buf, format="PNG")
            return buf.getvalue()
        studio.render_studio_portrait = fake_render

    def tearDown(self):
        (ls.detect_lip_points, beauty._save_download, studio.studio_render_available,
         studio.render_studio_portrait, beauty.restore_original_face) = self.orig

    def _save(self, **look):
        look = look or {"lipstick": LIP, "compact_powder": POWDER}
        return self.client.post("/api/beauty/tryon", json={"image": _b64(_portrait()), "include_qr": True, **look}).json()

    def _restore(self, restored, low_detail=False):
        from app.services.face_preservation import FaceCheck

        def fake(original, render_bytes, *_, **options):
            self.restore_options = options
            return render_bytes, FaceCheck(0.97, 0.95, restored, restored=restored, low_detail_skip=low_detail)
        beauty.restore_original_face = fake

    def test_studio_portrait_with_real_face(self):
        self._restore(True)
        res = self._save()
        self.assertTrue(res["success"], res)
        self.assertTrue(res["studio_render"])
        self.assertEqual(res["mode"], "full_makeover")
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Ruby Royale", self.calls[0])
        self.assertIn("Honey", self.calls[0])
        self.assertIn("SKIN TONE LOCK", self.calls[0])
        # The real face keeps its own skin tone and is barely enlarged (no lightening, no textureless lips)
        self.assertEqual(self.restore_options, {"keep_capture_tone": True, "max_upscale": studio.MAX_FACE_UPSCALE})

    def test_powder_only_studio_prompt(self):
        self._restore(True)
        res = self._save(compact_powder={**POWDER, "shade_id": "translucent", "shade_name": "Translucent"})
        self.assertTrue(res["studio_render"])
        self.assertIn("colorless", self.calls[0])
        self.assertIn("Lipstick: none", self.calls[0])

    def test_face_not_restorable_falls_back_to_local(self):
        self._restore(False)
        res = self._save()
        self.assertTrue(res["success"])
        self.assertFalse(res["studio_render"])
        self.assertTrue(res["qr_code_base64"])

    def test_low_detail_skip_is_not_a_real_face_and_falls_back(self):
        self._restore(False, low_detail=True)
        self.assertFalse(self._save()["studio_render"])

    def test_gemini_failure_falls_back_to_local(self):
        def boom(*a, **k):
            raise studio.StudioRenderError("quota")
        studio.render_studio_portrait = boom
        res = self._save()
        self.assertTrue(res["success"])
        self.assertFalse(res["studio_render"])

    def test_previews_never_call_gemini(self):
        self._restore(True)
        res = self.client.post("/api/beauty/tryon", json={"image": _b64(_portrait()), "lipstick": LIP,
                                                          "compact_powder": POWDER}).json()
        self.assertTrue(res["success"])
        self.assertEqual(self.calls, [])


class ClothingRoutesUntouchedTest(unittest.TestCase):
    def test_main_app_serves_both_pipelines(self):
        import main
        paths = set(main.app.openapi()["paths"])
        for path in ("/api/tryon", "/api/extract-profile", "/api/recommend-size", "/api/session/purge",
                     "/download/{image_id}", "/api/beauty/shades", "/api/beauty/powders", "/api/beauty/tryon"):
            self.assertIn(path, paths)


if __name__ == "__main__":
    unittest.main()
