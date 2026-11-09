"""
Measures everything in /api/tryon EXCEPT Gemini, at no cost: the Gemini client is replaced by a fake that returns a stored
render instantly, so the numbers are the local work (validation, pose, face restore, checks, encoding) on real code paths.

    venv\\Scripts\\python.exe tools\\local_pipeline_bench.py --render <render.jpg>
    venv\\Scripts\\python.exe tools\\local_pipeline_bench.py --render <render> --runs 5 --profile

    --render   a finished try-on image, ideally 3600x4800. Its downscaled copy stands in for the capture.
    --garment  catalog outfit id from frontend/src/assets/dress (default sherwani1)

Nothing is written to the real logs or the visitor records, and nothing leaves this PC.
"""
import argparse
import base64
import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import time
from types import SimpleNamespace

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRESS_DIR = os.path.join(BACKEND, "..", "..", "frontend", "src", "assets", "dress")
APPEARANCE_JSON = json.dumps({
    "gender_presentation": "Individual", "skin_tone": "as photographed", "own_sleeves": "full", "own_bottom": "separate",
    "wrist_accessories": "none",
})


def fake_client(render_bytes):
    def generate_content(model, contents, config=None):
        usage = SimpleNamespace(prompt_token_count=0, candidates_token_count=0, thoughts_token_count=0,
                                candidates_tokens_details=None)
        if getattr(config, "response_modalities", None):
            part = SimpleNamespace(text=None, inline_data=SimpleNamespace(data=render_bytes))
            return SimpleNamespace(parts=[part], candidates=[], usage_metadata=usage, prompt_feedback=None)
        return SimpleNamespace(text=APPEARANCE_JSON, parts=[], candidates=[], usage_metadata=usage, prompt_feedback=None)

    api_client = SimpleNamespace(api_key="fake", vertexai=False, _http_options=SimpleNamespace(base_url="fake"))
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content), _api_client=api_client)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", required=True)
    parser.add_argument("--garment", default="sherwani1")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="print the slowest functions of the last run")
    args = parser.parse_args()

    render_path, garment_path = os.path.abspath(args.render), os.path.join(DRESS_DIR, f"{args.garment}.png")
    os.chdir(BACKEND)
    sys.path.insert(0, BACKEND)
    os.environ.update(GEMINI_API_KEY="fake", GEMINI_COST_LOG="false", TRYON_TIMING_LOG="false", APP_LOG="false",
                      VISITOR_DATA_ENABLED="false")

    from PIL import Image
    with Image.open(render_path) as source:
        rgb = source.convert("RGB")
    buf = io.BytesIO()
    rgb.resize((3584, 4800), Image.Resampling.BICUBIC).save(buf, format="PNG", compress_level=1)  # Gemini-sized render
    render_bytes = buf.getvalue()
    buf = io.BytesIO()
    rgb.resize((540, 720), Image.Resampling.LANCZOS).save(buf, format="PNG")
    capture = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    with open(garment_path, "rb") as fh:
        dress = "data:image/png;base64," + base64.b64encode(fh.read()).decode()

    import main as app_main
    from fastapi.testclient import TestClient
    from app.routers import tryon
    from app.services import gemini_client, request_timing

    fake = fake_client(render_bytes)
    gemini_client.genai = SimpleNamespace(Client=lambda **_: fake)
    rows, finish = [], request_timing.finish

    def keep_row(timing, outcome, size, fit_category):
        row = finish(timing, outcome, size, fit_category)
        rows.append(row)
        return row
    request_timing.finish = keep_row

    stdout = sys.stdout
    try:
        with TestClient(app_main.app) as client:
            tryon.warm_up_yolo()
            profiler = cProfile.Profile()
            hits = []
            for run in range(args.runs + 1):
                session = f"sess_bench_{run}"
                # A new capture each run, so no capture-level cache carries over: one pixel differs
                img = Image.open(io.BytesIO(base64.b64decode(capture.split(",")[1]))).convert("RGB")
                img.putpixel((0, run), (run, run, run))
                b = io.BytesIO()
                img.save(b, format="PNG")
                run_capture = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
                payload = {"user_image": run_capture, "dress_image": dress, "selected_size": "L", "session_id": session}
                if args.profile and run == args.runs:
                    profiler.enable()
                sys.stdout = open(os.devnull, "w", encoding="utf-8")
                first = client.post("/api/tryon", json=payload)
                started = time.perf_counter()
                again = client.post("/api/tryon", json=payload)
                hits.append(time.perf_counter() - started)
                sys.stdout.close()
                sys.stdout = stdout
                if args.profile and run == args.runs:
                    profiler.disable()
                if first.status_code != 200 or again.status_code != 200:
                    sys.exit(f"Run {run}: HTTP {first.status_code}/{again.status_code}: {first.text[:300]}")
    finally:
        sys.stdout = stdout

    served = [r for r in rows if r["outcome"] == "served"][1:]  # the first run warms lazily loaded models
    if not served:
        sys.exit(f"No request was served. Outcomes: {[r['outcome'] for r in rows]}")
    print(f"\n{len(served)} generations, Gemini replaced by a stored {len(render_bytes) // 1024 // 1024} MB render "
          f"(attempts: {sorted(set(r['attempt_results'] for r in served))})")
    columns = [c for c in served[0] if c.endswith("_s")]
    for column in columns:
        values = [float(r[column]) for r in served]
        print(f"  {column:<18} median {statistics.median(values):6.2f}s   max {max(values):6.2f}s")
    print(f"  {'repeat request':<18} median {statistics.median(hits[1:]):6.2f}s   max {max(hits[1:]):6.2f}s   (same size and garment)")
    if args.profile:
        stats = pstats.Stats(profiler, stream=sys.stdout)
        stats.sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main()
