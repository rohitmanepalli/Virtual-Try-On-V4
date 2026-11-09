"""
End-to-end drape validation on real captures, through the real /api/extract-profile and /api/tryon endpoints.

Real run — paid Gemini calls (total printed at the end; every call is also in logs/gemini_costs.csv):
    venv\\Scripts\\python.exe tools\\drape_validation.py ^
        --person lean=calibration_captures\\capture_A.png ^
        --person standard=calibration_captures\\capture_B.png ^
        --person fuller=calibration_captures\\capture_C.png ^
        --garment ..\\frontend\\src\\assets\\dress\\male1.png --all-sizes fuller --force-flatten fuller

    Per person: profile (torso fullness / shape), render at the recommended size, full prompt and verifier result.
    --all-sizes P      also renders every other size for P (XS..XXL, "won't fit" included), as the result page does
                       on a size change, then a cross-size report: garment width vs the recommended render, near-identical
                       pairs, face match and skeleton drift across sizes. Repeatable.
    --force-flatten P  injects a "render slimmer" instruction into P's first attempt only, to prove the midsection
                       check fails it and the strict retry fires (one extra generation).
    Always (no API cost): the digital stomach test on the raw captures, with suggested shape bands / tolerance.

Wiring check — Gemini is faked, nothing is billed; prints the full prompts:
    venv\\Scripts\\python.exe tools\\drape_validation.py --dry-run --person standin=<any front photo> --all-sizes standin

Outputs (visitor photos — delete when testing ends) go to validation_runs/<timestamp>/.
"""
import argparse
import base64
import io
import json
import math
import os
import sys
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

_DRY_RUN = "--dry-run" in sys.argv
if _DRY_RUN:
    os.environ["GEMINI_COST_LOG"] = "false"  # config reads this at import; fake calls must not reach the cost CSV
    os.environ.setdefault("GEMINI_API_KEY", "dry-run")
else:
    from dotenv import load_dotenv
    load_dotenv()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance, ImageFont  # noqa: E402

from app.routers import tryon  # noqa: E402
from app.services import gemini_client  # noqa: E402
from app.services.body_profile import body_profile_service, classify_torso_shape  # noqa: E402
from app.services.gemini_cost import ANALYSIS_MODEL  # noqa: E402
from app.services.face_preservation import verify_face_preserved  # noqa: E402
from app.services.silhouette_verifier import (  # noqa: E402
    MIDSECTION_MAX_LOSS, compare_sizes, compare_to_recommended, garment_silhouette, silhouette_verifier,
)
from app.services.size_chart import ALL_SIZES  # noqa: E402

FLATTEN_INJECTION = (
    "\n[TEST ONLY — DELIBERATE FAILURE INJECTION]\n"
    "- Render this person noticeably slimmer than in Image 1: a flat stomach and a narrow, toned waist.\n"
)


def to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "png" if fmt == "PNG" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def from_data_uri(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


# ── Instrumentation: record every attempt's prompt, output, verifier result and cost ─────────────────────────────
class Recorder:
    def __init__(self):
        self.attempts = []  # one dict per Gemini generation attempt
        self.cost = 0.0
        self.calls = []
        self.inject_flatten_next = False
        self.force_flatten = False  # set per person; injects into attempt 1 of the next request
        self._real_generate = tryon.generate_tryon_image
        self._real_build = gemini_client.build_tryon_prompt
        self._real_verify = silhouette_verifier.verify
        self._real_record = gemini_client.record_usage

        self._real_eval = tryon._evaluate_fit

    def patches(self):
        return [
            patch.object(tryon, "generate_tryon_image", self.generate),
            patch.object(gemini_client, "build_tryon_prompt", self.build),
            patch.object(silhouette_verifier, "verify", self.verify),
            patch.object(gemini_client, "record_usage", self.record),
            patch.object(tryon, "_evaluate_fit", self.evaluate_fit),
        ]

    def evaluate_fit(self, metrics, *a, **kw):
        result = self._real_eval(metrics, *a, **kw)
        if self.attempts:
            self.attempts[-1]["garment"] = {k: v for k, v in (metrics or {}).items() if k != "appearance"}
            self.attempts[-1]["fit_eval"] = {"passed": result["passed"], "checked": result["checked"],
                                             "problem": result["problem"], "other_size": result["other_size"],
                                             "direction": result["direction"], "pairs": result["pairs"]}
        return result

    def generate(self, **kw):
        self.attempts.append({"size": kw.get("selected_size"), "attempt": kw.get("attempt"),
                              "retry_strict": kw.get("retry_strict"), "fit_escalation": kw.get("fit_escalation") is not None,
                              "fit": getattr(kw.get("size_offset"), "fit_quality", None),
                              "offset": getattr(kw.get("size_offset"), "steps_from_recommended", None),
                              "face_lock_source": kw.get("identity_face") is not None})
        if self.force_flatten and kw.get("attempt") == 1:
            self.inject_flatten_next = True
        result = self._real_generate(**kw)
        self.attempts[-1]["image"] = Image.open(io.BytesIO(result[0])).convert("RGB")
        return result

    def build(self, **kw):
        prompt = self._real_build(**kw)
        if self.inject_flatten_next:
            prompt += FLATTEN_INJECTION
            self.inject_flatten_next = False
            self.attempts[-1]["flatten_injected"] = True
        if self.attempts:
            self.attempts[-1]["prompt"] = prompt
        return prompt

    def verify(self, **kw):
        result = self._real_verify(**kw)
        if self.attempts:
            self.attempts[-1]["verify"] = {"passed": result.passed, "score": result.score, **(result.details or {})}
        return result

    def record(self, kind, model, response, **kw):
        cost = self._real_record(kind, model, response, **kw)
        self.calls.append({"kind": kind, "model": model, "cost": cost, **kw})
        self.cost += cost or 0.0
        return cost


# ── Dry run: a fake Gemini client that returns the capture as the "render" so every real stage still runs ───────
class _FakeModels:
    def generate_content(self, model, contents, config=None):
        prompt = contents[0]
        if model == ANALYSIS_MODEL:
            if "clothing item" in prompt:
                data = {"category": "shirt", "garment_name": "Dry-run shirt", "fabric_type": "Woven cotton",
                        "structure_level": "semi_structured", "is_outerwear": False,
                        "sleeve_behavior": "Clean sleeve drape", "tryon_guidelines": "Dry-run guidelines"}
            else:
                data = {"gender_presentation": "(dry run)", "skin_tone": "(dry run)", "body_build": "(dry-run build)",
                        "midsection": "(dry-run midsection)"}
            return SimpleNamespace(text=json.dumps(data), usage_metadata=None)
        buf = io.BytesIO()
        contents[1].save(buf, format="PNG")
        part = SimpleNamespace(text=None, inline_data=SimpleNamespace(data=buf.getvalue()))
        return SimpleNamespace(parts=[part], candidates=[], prompt_feedback=None, usage_metadata=None,
                               sdk_http_response=None)


class _FakeClient:
    def __init__(self, *a, **kw):
        self.models = _FakeModels()
        self._api_client = SimpleNamespace(api_key="dry-run", vertexai=False, _http_options=SimpleNamespace(base_url=None))


# ── Digital stomach test on raw captures (no API cost) ───────────────────────────────────────────────────────────
def _reframe(img, s, dx=0.0):
    w, h = img.size
    im = img.resize((int(w * s), int(h * s)))
    corner = img.getpixel((5, 5))
    canvas = Image.new("RGB", (w, h), corner)
    canvas.paste(im, (int((w - im.width) / 2 + dx * w), int((h - im.height) / 2)))
    return canvas


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


PERTURB = {
    "scale0.85": lambda im: _reframe(im, 0.85), "shift+4%": lambda im: _reframe(im, 0.9, 0.04),
    "shift-4%": lambda im: _reframe(im, 0.9, -0.04), "jpeg60": lambda im: _jpeg(im, 60),
    "dark0.8": lambda im: ImageEnhance.Brightness(im).enhance(0.8),
}


def _torso_edges(img):
    """Per image row between shoulder and hip lines: torso edges (image px), using the production arm exclusion."""
    f = body_profile_service.extract_single_frame(img)
    kp = f.keypoints
    person = np.squeeze(f.segmentation_mask) > 0.5
    mh, mw = person.shape
    iw, ih = img.size
    l_sh, r_sh, l_hip, r_hip = kp["left_shoulder"], kp["right_shoulder"], kp["left_hip"], kp["right_hip"]
    sh_y, hip_y = (l_sh[1] + r_sh[1]) / 2, (l_hip[1] + r_hip[1]) / 2
    cx = (l_sh[0] + r_sh[0] + l_hip[0] + r_hip[0]) / 4
    arms = sorted(([l_sh, kp.get("left_elbow"), kp.get("left_wrist")], [r_sh, kp.get("right_elbow"), kp.get("right_wrist")]),
                  key=lambda c: c[0][0])
    rows, lefts, rights = [], [], []
    for y_px in range(int(sh_y * ih), int(hip_y * ih)):
        y_n = y_px / ih
        run = body_profile_service._silhouette_run(person, y_n, cx)
        if run is None:
            continue
        edges = [float(run[0]), float(run[1])]
        for i, arm in enumerate(arms):
            ax = body_profile_service._arm_center_x(arm, y_n)
            if ax is not None:
                apx, outer, center = ax * (mw - 1), edges[i], cx * (mw - 1)
                if min(outer, center) < apx < max(outer, center):
                    edges[i] = 2 * apx - outer
        rows.append(y_px)
        lefts.append(edges[0] / mw * iw)
        rights.append(edges[1] / mw * iw)
    k = np.ones(15) / 15
    lefts = np.convolve(np.pad(lefts, 7, mode="edge"), k, mode="valid")
    rights = np.convolve(np.pad(rights, 7, mode="edge"), k, mode="valid")
    return np.array(rows), lefts, rights, sh_y * ih, hip_y * ih, cx * iw


def flatten_torso(img, amp):
    """Narrow the torso by up to `amp` at the stomach (zero at shoulders and hips); arms move in rigidly."""
    rows, lefts, rights, sh_y, hip_y, cx = _torso_edges(img)
    w, h = img.size
    map_x = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w)).copy()
    map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w)).copy()
    xs = np.arange(w, dtype=np.float32)
    for y, el, er in zip(rows, lefts, rights):
        t = (y - sh_y) / max(1.0, hip_y - sh_y)
        bump = np.sin(np.pi / 2 * t / 0.65) ** 2 if t <= 0.65 else np.sin(np.pi / 2 * (1 - t) / 0.35) ** 2
        s = 1.0 - amp * bump
        el_new, er_new = cx + (el - cx) * s, cx + (er - cx) * s
        row = xs.copy()
        inside = (xs >= el_new) & (xs <= er_new)
        row[inside] = cx + (xs[inside] - cx) / s
        row[xs < el_new] = xs[xs < el_new] - (el_new - el)
        row[xs > er_new] = xs[xs > er_new] + (er - er_new)
        map_x[y] = row
    return Image.fromarray(cv2.remap(np.array(img), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def stomach_test(people):
    """Fullness separation, repeatability, false alarms and flattening detection on the raw captures."""
    print("\n=== Digital stomach test on raw captures (no API calls) ===")
    out = {}
    for label, img in people.items():
        base = body_profile_service.extract_single_frame(img)
        full = body_profile_service.torso_fullness(base)
        if full is None:
            print(f"  {label}: torso fullness NOT measurable on this capture")
            out[label] = {"fullness": None}
            continue
        pert = [body_profile_service.torso_fullness(body_profile_service.extract_single_frame(fn(img))) for fn in PERTURB.values()]
        pert = [p for p in pert if p is not None]
        drops = [p / full - 1 for p in pert]
        detect = {}
        for amp in (0.10, 0.15, 0.20, 0.25):
            r = silhouette_verifier.check_midsection(base, body_profile_service.extract_single_frame(flatten_torso(img, amp)))
            detect[amp] = r["change"] if r else None
        out[label] = {"fullness": round(full, 3), "shape": classify_torso_shape(full), "worst_unchanged_drop": round(min(drops), 3),
                      "cv": round(float(np.std([full] + pert) / np.mean([full] + pert)), 3),
                      "flatten_change": {f"{int(a * 100)}%": c for a, c in detect.items()}}
        print(f"  {label:10s} fullness {full:.3f} ({classify_torso_shape(full)}) | repeatability cv {out[label]['cv']:.1%}, "
              f"worst unchanged drop {min(drops):+.1%} | flatten 10/15/20/25% -> "
              + " / ".join(f"{c:+.1%}" if c is not None else "n/a" for c in detect.values()))

    measured = {k: v["fullness"] for k, v in out.items() if v.get("fullness") is not None}
    order = [k for k in ("lean", "standard", "fuller") if k in measured]
    if len(order) >= 2:
        vals = [measured[k] for k in order]
        ordered = all(a < b for a, b in zip(vals, vals[1:]))
        print(f"  Order lean < standard < fuller: {'YES' if ordered else 'NO — the score does not separate these people as labelled'}")
        if ordered:
            mids = [round((a + b) / 2, 3) for a, b in zip(vals, vals[1:])]
            print(f"  Suggested band boundaries between labelled people: {mids} (current TORSO_SHAPE_BANDS upper bounds 0.92 / 1.02 / 1.15)")
    worst = min((v["worst_unchanged_drop"] for v in out.values() if v.get("fullness") is not None), default=None)
    if worst is not None:
        print(f"  Worst unchanged-body drop {worst:+.1%}; current tolerance {-MIDSECTION_MAX_LOSS:+.0%} "
              f"({'margin OK' if worst > -MIDSECTION_MAX_LOSS + 0.01 else 'TOO TIGHT — raise the tolerance'})")
    return out


# ── Contact sheets ────────────────────────────────────────────────────────────────────────────────────────────────
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def sheet(tiles, path, tile_h=900):
    """tiles: list of (image or None, caption lines)."""
    font = _font(26)
    scaled = []
    for img, lines in tiles:
        if img is None:
            img = Image.new("RGB", (int(tile_h * 0.75), tile_h), (235, 235, 235))
        img = img.resize((int(img.width * tile_h / img.height), tile_h))
        scaled.append((img, lines))
    cap_h = 40 + 32 * max(len(lines) for _, lines in scaled)
    canvas = Image.new("RGB", (sum(i.width for i, _ in scaled) + 20 * (len(scaled) + 1), tile_h + cap_h + 20), "white")
    draw = ImageDraw.Draw(canvas)
    x = 20
    for img, lines in scaled:
        canvas.paste(img, (x, 10))
        for j, line in enumerate(lines):
            draw.text((x, tile_h + 20 + 32 * j), line, fill=(20, 20, 20), font=font)
        x += img.width + 20
    canvas.save(path, quality=92)
    print(f"  sheet: {path}")


# ── Cross-size report (no API cost): garment width vs recommended, near-identical pairs, face and body lock ──────────
def cross_size_report(label, renders, person, recommended):
    print(f"\n=== {label}: cross-size verification (served renders) ===")
    qualities = {s: (person["renders"].get(s) or {}).get("fit_quality") for s in ALL_SIZES}
    metrics = {}
    for s, img in renders.items():
        if img is not None:
            metrics[s] = garment_silhouette(img, body_profile_service.extract_single_frame(img))
    rec_img, rec_m = renders.get(recommended), metrics.get(recommended)
    out = {"sizes": {}, "pairs": {}}
    for s in ALL_SIZES:
        m = metrics.get(s)
        row = {"torso_width": (m or {}).get("torso_width"), "outer_width": (m or {}).get("outer_width")}
        if m and rec_m and s != recommended:
            d = compare_to_recommended(m, rec_m, qualities.get(s) or "perfect")
            row.update(torso_delta=d["torso_delta"], outer_delta=d["outer_delta"], direction_ok=d["passed"])
        if rec_img is not None and renders.get(s) is not None and s != recommended:
            fc = verify_face_preserved(rec_img, renders[s])
            row["face"] = f"pixel {fc.pixel_similarity:.3f} ssim {fc.ssim:.3f}" + (" (identical)" if fc.passed else " (DIFFERENT)")
        out["sizes"][s] = row
        print(f"  {s:4s} {str(qualities.get(s)):13s} torso {row['torso_width']} outer {row['outer_width']} | "
              f"vs rec torso {row.get('torso_delta')} outer {row.get('outer_delta')} direction ok {row.get('direction_ok')} | "
              f"face vs rec: {row.get('face', 'n/a')}")
    sizes = [s for s in ALL_SIZES if metrics.get(s)]
    near = []
    for i, a in enumerate(sizes):
        for b in sizes[i + 1:]:
            c = compare_sizes(metrics[a], qualities.get(a) or "perfect", metrics[b], qualities.get(b) or "perfect")
            out["pairs"][f"{a}-{b}"] = c
            if c["near_identical"]:
                near.append(f"{a}-{b}")
    print("  pairwise silhouette distance: " + ", ".join(f"{k} {v['silhouette_distance']}" for k, v in out["pairs"].items()))
    print(f"  near-identical pairs: {near or 'none'}")
    served = {s: img for s, img in renders.items() if img is not None}
    if len(served) >= 2:
        passed, details = silhouette_verifier.verify_cross_size_consistency(served)
        out["body_consistency"] = {"max_body_drift": details.get("max_body_drift"), "body_constant": details.get("body_constant")}
        print(f"  body skeleton drift across sizes: max {details.get('max_body_drift')} (constant: {details.get('body_constant')}, limit 0.08)")
    out["near_identical"] = near
    return out


# ── Pipeline driving ──────────────────────────────────────────────────────────────────────────────────────────────
def run(args):
    people = {}
    for spec in args.person:
        label, path = spec.split("=", 1)
        people[label] = Image.open(path).convert("RGB")
    garment = Image.open(args.garment).convert("RGB")
    run_dir = os.path.join("validation_runs", datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dryrun" if args.dry_run else ""))
    os.makedirs(os.path.join(run_dir, "prompts"), exist_ok=True)
    print(f"Run folder: {os.path.abspath(run_dir)}  ({'DRY RUN — Gemini faked, nothing billed' if args.dry_run else 'REAL Gemini calls'})")

    app = FastAPI()
    app.include_router(tryon.router)
    client = TestClient(app)
    rec = Recorder()
    summary = {"dry_run": args.dry_run, "garment": args.garment, "people": {}}

    with ExitStack() as stack:
        for p in rec.patches():
            stack.enter_context(p)
        if args.dry_run:
            stack.enter_context(patch.object(gemini_client.genai, "Client", _FakeClient))

        garment_uri = to_data_uri(garment)
        all_sizes_for = set(args.all_sizes or [])
        for label, capture in people.items():
            sid = f"validation-{label}-{datetime.now().strftime('%H%M%S')}"
            prof = client.post("/api/extract-profile", json={"image": to_data_uri(capture), "hold_frames": [], "session_id": sid}).json()
            bp = prof["body_profile"]
            print(f"\n=== {label}: fullness {bp.get('torso_fullness')} ({bp.get('torso_shape')}), body_type '{bp['body_type']}' ===")
            person = {"profile": {k: bp.get(k) for k in ("torso_fullness", "torso_shape", "body_type", "chest_proportion",
                                                         "waist_proportion", "shoulder_width_ratio")}, "renders": {}}
            rec.force_flatten = label == args.force_flatten
            renders, last_good, recommended = {}, None, None
            sizes = [None] + (list(ALL_SIZES) if label in all_sizes_for else [])
            for size in sizes:
                if size is not None and size == recommended:
                    continue
                n_before, cost_before = len(rec.attempts), rec.cost
                resp = client.post("/api/tryon", json={
                    "user_image": to_data_uri(capture), "dress_image": garment_uri, "selected_size": size,
                    "body_profile": bp, "session_id": sid, "last_good_render": last_good,
                })
                data = resp.json()
                attempts = rec.attempts[n_before:]
                served_size = data.get("selected_size") or size or "auto"
                for a in attempts:
                    tag = f"{label}_{served_size}_attempt{a['attempt']}"
                    with open(os.path.join(run_dir, "prompts", tag + ".txt"), "w", encoding="utf-8") as fh:
                        fh.write(a.get("prompt", ""))
                    if a.get("image") is not None:
                        a["image"].save(os.path.join(run_dir, tag + ".jpg"), quality=90)
                entry = {
                    "fit_quality": data.get("size_fit_quality"), "fit_label": data.get("fit_label"),
                    "size_offset": data.get("size_offset"), "wont_fit": data.get("wont_fit"),
                    "illustration_only": data.get("illustration_only"), "fit_verified": data.get("fit_verified"),
                    "fit_check": data.get("fit_check"), "message": data.get("message"),
                    "silhouette_score": data.get("silhouette_score"), "cost_usd": round(rec.cost - cost_before, 4),
                    "attempts": [{k: v for k, v in a.items() if k not in ("image", "prompt")} for a in attempts],
                }
                person["renders"][served_size] = entry
                rec.force_flatten = False
                if recommended is None and data.get("size_recommendation"):
                    recommended = data["size_recommendation"]["recommended_size"]
                    person["size_fits"] = data["size_recommendation"].get("size_fits")
                for a in attempts:
                    v, g, fe = a.get("verify", {}), a.get("garment") or {}, a.get("fit_eval") or {}
                    d = fe.get("direction") or {}
                    print(f"  size {served_size} ({a.get('fit')}, offset {a.get('offset')}) attempt {a['attempt']}"
                          f"{' [flatten injected]' if a.get('flatten_injected') else ''}{' [strict retry]' if a.get('retry_strict') else ''}"
                          f"{' [fit retry]' if a.get('fit_escalation') else ''}: midsection {v.get('midsection_change', float('nan')):+.1%} "
                          f"preserved {v.get('midsection_preserved')}, score {v.get('score')} | garment torso {g.get('torso_width')} "
                          f"outer {g.get('outer_width')} | vs rec torso {d.get('torso_delta')} outer {d.get('outer_delta')} | "
                          f"fit check {'PASS' if fe.get('passed') else 'FAIL ' + str(fe.get('problem')) + ' ' + str(fe.get('other_size'))}")
                if resp.status_code != 200 or not data.get("result_image"):
                    state = ("WON'T FIT: message + illustration (" + str((data.get("fit_check") or {}).get("render_rejected")) + ")"
                             if data.get("illustration_only") else f"NO IMAGE (HTTP {resp.status_code}): {data.get('detail') or data.get('message')}")
                    print(f"  size {served_size}: {state} | cost ${rec.cost - cost_before:.3f}")
                    renders[served_size] = None
                    continue
                img = from_data_uri(data["result_image"])
                img.save(os.path.join(run_dir, f"{label}_{served_size}.jpg"), quality=92)
                renders[served_size] = img
                if not data.get("wont_fit"):
                    last_good = data["result_image"]
                print(f"  size {served_size}: served ({data.get('fit_label')}, offset {data.get('size_offset')}, "
                      f"fit_verified {data.get('fit_verified')}){' — ' + data['message'] if data.get('wont_fit') else ''} | "
                      f"{len(attempts)} attempt(s), cost ${rec.cost - cost_before:.3f}")

            person["recommended_size"] = recommended
            summary["people"][label] = person
            if label in all_sizes_for:
                person["cross_size"] = cross_size_report(label, renders, person, recommended)
                tiles = [(capture, [f"{label} capture", f"shape {bp.get('torso_shape')} ({bp.get('torso_fullness')})"])]
                for s in ALL_SIZES:
                    info = person["renders"].get(s, {})
                    cs = person["cross_size"]["sizes"].get(s, {})
                    tiles.append((renders.get(s), [
                        f"{s}{' (recommended)' if s == recommended else ''}  offset {info.get('size_offset'):+d}" if info.get('size_offset') is not None else s,
                        f"{info.get('fit_label')}" + (" — ILLUSTRATION ONLY" if info.get("illustration_only") else ""),
                        f"garment vs rec: torso {cs.get('torso_delta')} outer {cs.get('outer_delta')}",
                        f"face vs rec: {cs.get('face')}",
                        f"{len(info.get('attempts', []))} attempt(s), ${info.get('cost_usd', 0):.3f}",
                    ]))
                sheet(tiles, os.path.join(run_dir, f"sizes_{label}.jpg"))
            if label == args.force_flatten:
                first = [a for a in rec.attempts if a.get("flatten_injected")]
                if first:
                    idx = rec.attempts.index(first[-1])
                    pair = rec.attempts[idx:idx + 2]
                    sheet([(a.get("image"), [f"attempt {a['attempt']}{' (flatten injected)' if a.get('flatten_injected') else ' (retry)'}",
                                             f"midsection {a.get('verify', {}).get('midsection_change', float('nan')):+.1%}",
                                             f"preserved {a.get('verify', {}).get('midsection_preserved')}"]) for a in pair],
                          os.path.join(run_dir, f"force_flatten_{label}.jpg"))
            person["_render_rec"] = renders.get(recommended)

        # Same garment, each person at their recommended size
        tiles = []
        for label, person in summary["people"].items():
            tiles.append((people[label], [f"{label} capture", f"fullness {person['profile']['torso_fullness']} ({person['profile']['torso_shape']})"]))
            tiles.append((person.pop("_render_rec"), [f"{label} render, size {person['recommended_size']}", person["profile"]["body_type"]]))
        sheet(tiles, os.path.join(run_dir, "side_by_side_recommended.jpg"))

    first_prompt = next((a for a in rec.attempts if a.get("prompt")), None)
    if first_prompt:
        print("\n=== FULL FINAL PROMPT — first generation of this run ===\n" + first_prompt["prompt"] + "\n=== END PROMPT ===")

    summary["stomach_test"] = stomach_test(people)
    summary["gemini_calls"] = len(rec.calls)
    summary["total_cost_usd"] = round(rec.cost, 4)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nGemini calls: {len(rec.calls)} | total cost: ${rec.cost:.3f}{' (dry run: nothing billed)' if args.dry_run else ''}")
    print(f"Summary: {os.path.abspath(os.path.join(run_dir, 'summary.json'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", action="append", required=True, help="label=path to a raw front capture (repeatable)")
    parser.add_argument("--garment", default=os.path.join("..", "frontend", "src", "assets", "dress", "male1.png"))
    parser.add_argument("--all-sizes", action="append", default=None, help="label of a person to render in every size (repeatable)")
    parser.add_argument("--force-flatten", default=None, help="label of the person whose first attempt is told to slim the body")
    parser.add_argument("--dry-run", action="store_true", help="fake Gemini: wiring check only, nothing billed")
    run(parser.parse_args())
