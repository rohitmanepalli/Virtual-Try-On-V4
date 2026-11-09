"""
Times one full visitor flow and prints how long each step takes, so latency changes can be checked before/after.

COSTS REAL MONEY: every run makes real Gemini calls on the GEMINI_API_KEY in backend/.env, about $0.15 per image
generation attempt. A normal run is 2 generations (first render + one size change) ~ $0.31; with retries (up to 3
attempts per render) up to ~ $0.92, plus a cheap appearance analysis. Every call is logged in logs/gemini_costs.csv.

    venv\\Scripts\\python.exe tools\\latency_probe.py --capture calibration_captures\\capture_X.png
    venv\\Scripts\\python.exe tools\\latency_probe.py --capture <photo> --garment kurta1 --no-size-change --yes

    --capture   a kiosk capture (612x720 front photo, as the capture screen sends it)
    --garment   catalog outfit id from frontend/src/assets/dress (default sherwani1) or a path to a garment PNG
    --no-size-change   only the first render (one generation fewer)
    --yes       skip the cost confirmation

Flow, through the real app in this process (the live server on :8000 is not touched): check-face -> extract-profile ->
recommend-size -> first try-on -> size change one size up. Models are warmed first, as on a running kiosk; this
visitor's caches (appearance analysis, face mask) start cold, as for a new visitor.

Prints: each request's total time, the nested step tree (call order, durations), and the same per-step columns that
every real request writes to logs/tryon_timings.csv. Probe requests are not written to that log, and download copies
go to a temporary folder that is deleted afterwards. The browser-side steps (5 s countdown, photo encoding, the
generating screen's 4.5 s minimum) are not part of this measurement. A busy kiosk server running at the same time
slows the local steps.
"""
import argparse
import base64
import functools
import os
import shutil
import sys
import tempfile
import threading
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRESS_DIR = os.path.join(BACKEND, "..", "frontend", "src", "assets", "dress")

# Steps shown in the tree: (module, attribute). Names that no longer exist are reported and skipped.
TRACED = {
    "app.routers.tryon": [
        "clean_and_decode_base64", "_validate_person", "_validate_body", "prepare_user_portrait", "analyze_cloth",
        "_is_studio_render", "analyze_user_appearance", "recommend_size", "size_fit_map", "get_size_offset",
        "_segment_person", "_read_capture_coverage", "_capture_has_face", "generate_tryon_image", "garment_silhouette",
        "_evaluate_fit", "_check_coverage", "_finalize_render", "_face_score", "finalize_result_image",
        "apply_watermark", "generate_qr_data_uri",
    ],
    "app.services.gemini_client": ["build_tryon_prompt", "restore_original_face"],
}
TRACED_METHODS = {
    "body_profile_service": ["extract_frames", "profile_from_frames", "measure_body", "extract_profile",
                             "extract_single_frame", "generate_face_and_hands_mask"],
    "silhouette_verifier": ["verify", "verify_cross_fit_landmarks"],
}


class Tracer:
    """Records nested call durations per request phase."""

    def __init__(self):
        self.records, self.phase, self._local = [], "setup", threading.local()

    def wrap(self, name, fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            stack = self._local.__dict__.setdefault("stack", [])
            depth = len(stack)
            stack.append(name)
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                stack.pop()
                self.records.append((self.phase, depth, name, time.perf_counter() - started, started))
        return inner

    def tree(self, phase):
        """(depth, name, seconds, count) in call order, consecutive identical calls merged."""
        rows = []
        for _, depth, name, secs, _ in sorted((r for r in self.records if r[0] == phase), key=lambda r: r[4]):
            if rows and rows[-1][:2] == (depth, name):
                rows[-1] = (depth, name, rows[-1][2] + secs, rows[-1][3] + 1)
            else:
                rows.append((depth, name, secs, 1))
        return rows


def install(tracer):
    import importlib
    from google.genai import models as genai_models

    missing = []
    for module_name, names in TRACED.items():
        module = importlib.import_module(module_name)
        for name in names:
            if hasattr(module, name):
                setattr(module, name, tracer.wrap(name, getattr(module, name)))
            else:
                missing.append(f"{module_name}.{name}")
    from app.services.body_profile import body_profile_service
    from app.services.silhouette_verifier import silhouette_verifier
    for label, obj in (("body_profile_service", body_profile_service), ("silhouette_verifier", silhouette_verifier)):
        for name in TRACED_METHODS[label]:
            if hasattr(obj, name):
                setattr(obj, name, tracer.wrap(f"{label.split('_')[0]}.{name}", getattr(obj, name)))
            else:
                missing.append(f"{label}.{name}")
    original = genai_models.Models.generate_content

    def generate_content(self, *args, **kwargs):
        return tracer.wrap(f"GEMINI {kwargs.get('model')}", original)(self, *args, **kwargs)
    genai_models.Models.generate_content = generate_content
    if missing:
        print(f"[probe] not traced (renamed or removed): {', '.join(missing)}")


def data_url(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--garment", default="sherwani1")
    parser.add_argument("--no-size-change", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    garment = args.garment if os.path.isfile(args.garment) else os.path.join(DRESS_DIR, f"{args.garment}.png")
    for label, path in (("capture", args.capture), ("garment", garment)):
        if not os.path.isfile(path):
            sys.exit(f"{label} not found: {path}")
    generations = 1 if args.no_size_change else 2
    print(f"This run makes REAL Gemini calls: {generations} image generation(s) at ~$0.15 each, up to "
          f"{3 * generations} with retries (~ ${0.154 * generations:.2f}-${0.154 * 3 * generations:.2f}), "
          f"plus one cheap appearance analysis. All are logged in logs/gemini_costs.csv.")
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")

    os.chdir(BACKEND)
    sys.path.insert(0, BACKEND)
    os.environ["CALIBRATION_CAPTURE_SAVE"] = "false"   # no visitor photo copies from the probe
    os.environ["MEASUREMENT_CORRECTION_LOG"] = "false"

    import main as app_main
    from fastapi.testclient import TestClient
    from app.routers import tryon
    from app.services import image_utils, request_timing

    scratch = tempfile.mkdtemp(prefix="latency_probe_")
    request_timing.TRYON_TIMING_LOG = False              # probe rows stay out of the real-usage timing log
    timing_rows = {}
    finish = request_timing.finish

    def capture_row(timing, outcome, size, fit_category):
        timing_rows[tracer.phase] = finish(timing, outcome, size, fit_category)
        return timing_rows[tracer.phase]
    request_timing.finish = capture_row

    tracer = Tracer()
    install(tracer)
    capture, dress = data_url(args.capture), data_url(garment)
    session = f"sess_latency_probe_{int(time.time())}"
    totals = []

    try:
        with TestClient(app_main.app) as client:
            print("[probe] warming models (YOLO, segmentation, MediaPipe)...", flush=True)
            tracer.phase = "warmup"
            tryon.warm_up_yolo()
            tryon._get_yolo_segmentation_service()
            client.post("/api/check-face", json={"image": capture})
            client.post("/api/extract-profile", json={"image": capture, "hold_frames": [capture, capture],
                                                      "session_id": "sess_probe_warmup"})

            def call(label, path, payload):
                tracer.phase = label
                print(f"[probe] {label}...", flush=True)
                started = time.perf_counter()
                resp = client.post(path, json=payload)
                totals.append((label, time.perf_counter() - started, resp.status_code))
                return resp

            call("check-face (capture screen)", "/api/check-face", {"image": capture})
            profile = call("extract-profile (background after capture)", "/api/extract-profile",
                           {"image": capture, "hold_frames": [capture, capture], "session_id": session}).json()
            body_profile = profile.get("body_profile")
            call("recommend-size (outfit page)", "/api/recommend-size",
                 {"session_id": session, "body_profile": body_profile, "garment_category": ""})
            first = call("try-on, first render (generating screen)", "/api/tryon",
                         {"user_image": capture, "dress_image": dress, "selected_size": None, "body_profile": body_profile,
                          "session_id": session, "last_good_render": None}).json()
            if not args.no_size_change and first.get("result_image"):
                size = first.get("selected_size")
                sizes = (first.get("size_recommendation") or {}).get("available_sizes") or []
                if size in sizes:
                    i = sizes.index(size)
                    other = sizes[i + 1] if i + 1 < len(sizes) else sizes[i - 1]
                    call(f"try-on, size change {size}->{other} (result screen)", "/api/tryon",
                         {"user_image": capture, "dress_image": dress, "selected_size": other,
                          "body_profile": body_profile, "session_id": session,
                          "last_good_render": first.get("result_image")})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print("\n" + "=" * 78 + "\nREQUESTS")
    for label, secs, status in totals:
        print(f"  {label:<52} {secs:7.2f}s  HTTP {status}")
    for label, secs, _ in totals:
        print(f"\n--- {label}: {secs:.2f}s")
        for depth, name, step_secs, count in tracer.tree(label):
            print(f"  {'   ' * depth}{name:<{50 - 3 * depth}}{step_secs:7.2f}s" + (f"  (x{count})" if count > 1 else ""))
        traced = sum(s for d, _, s, _ in tracer.tree(label) if d == 0)
        print(f"  {'(rest: JSON/base64, sessions, small logic)':<50}{secs - traced:7.2f}s")
        row = timing_rows.get(label)
        if row:
            cols = [c for c in row if c.endswith("_s") and c != "total_s"]
            print("  as logged by real requests (logs/tryon_timings.csv columns): "
                  + ", ".join(f"{c[:-2]} {float(row[c]):.1f}s" for c in cols)
                  + f" | attempts: {row['attempt_results'] or '-'} | outcome: {row['outcome']}")
    print("\nNot measured here (browser): 5 s countdown, ~1 s photo encoding, 4.5 s minimum on the generating screen.")


if __name__ == "__main__":
    main()
