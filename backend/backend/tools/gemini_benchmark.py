"""
Compares Gemini image models and resolutions through the real /api/tryon pipeline, so each result includes the real face
restore and validation checks. Prints latency, cost and check results, and saves every output for a side-by-side look:
speed alone can't show whether identity and garment detail held up.

COSTS REAL MONEY on the GEMINI_API_KEY in backend/.env (about $0.15 per 4K image, less at lower resolutions).
It asks first and prints the estimate. Calls are logged in logs/gemini_costs.csv like any others.

    venv\\Scripts\\python.exe tools\\gemini_benchmark.py --list-models
    venv\\Scripts\\python.exe tools\\gemini_benchmark.py --capture <photo> --out <folder> --runs 3 \
        --configs gemini-3.1-flash-image:4K,gemini-3.1-flash-image:2K

    --capture  a kiosk capture. Sent to Google, like any try-on.
    --out      folder for the outputs (holds images of the person: delete it when done)

Rounds run in turn across the configs, so a slow patch of the API doesn't land on one config. One attempt per run: a
failed check is reported instead of paying for a retry.
"""
import argparse
import base64
import csv
import os
import statistics
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRESS_DIR = os.path.join(BACKEND, "..", "..", "frontend", "src", "assets", "dress")
# List prices: output image tokens x $ per 1M, for the estimate shown before running
ESTIMATE_USD = {"4K": 0.154, "2K": 0.103, "1K": 0.069}


def data_url(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture")
    parser.add_argument("--garment", default="sherwani1")
    parser.add_argument("--configs", default="gemini-3.1-flash-image:4K,gemini-3.1-flash-image:2K")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    capture_path = os.path.abspath(args.capture) if args.capture else None
    out_dir = os.path.abspath(args.out) if args.out else None
    garment_path = os.path.join(DRESS_DIR, f"{args.garment}.png")
    os.chdir(BACKEND)
    sys.path.insert(0, BACKEND)
    from dotenv import load_dotenv
    load_dotenv()

    if args.list_models:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # must outlive the listing, which closes it when collected
        for model in client.models.list():
            if "image" in model.name.lower():
                print(model.name, getattr(model, "display_name", ""))
        return

    if not (capture_path and out_dir):
        sys.exit("--capture and --out are required.")
    configs = [tuple(c.split(":")) for c in args.configs.split(",")]
    estimate = sum(ESTIMATE_USD.get(size, 0.154) for _, size in configs) * args.runs
    print(f"{len(configs) * args.runs} real generations, about ${estimate:.2f}.")
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")

    os.environ["APP_LOG"] = "false"
    os.environ["VISITOR_DATA_ENABLED"] = "false"
    os.environ["CALIBRATION_CAPTURE_SAVE"] = "false"
    os.environ["MEASUREMENT_CORRECTION_LOG"] = "false"
    os.makedirs(out_dir, exist_ok=True)

    import main as app_main
    from fastapi.testclient import TestClient
    from app.routers import tryon
    from app.services import gemini_client, request_timing

    tryon.GENERATION_MAX_ATTEMPTS = 1
    request_timing.TRYON_TIMING_LOG = False

    timing_rows, finish = [], request_timing.finish

    def keep_row(timing, outcome, size, fit_category):
        timing_rows.append(finish(timing, outcome, size, fit_category))
        return timing_rows[-1]
    request_timing.finish = keep_row

    usage, record = [], gemini_client.record_usage

    def keep_usage(kind, model, response, **kwargs):
        meta = getattr(response, "usage_metadata", None)
        if kind == "tryon_generation":
            usage.append({"cost": record(kind, model, response, **kwargs), "tokens": getattr(meta, "candidates_token_count", None),
                          "seconds": kwargs.get("duration_s")})
            return usage[-1]["cost"]
        return record(kind, model, response, **kwargs)
    gemini_client.record_usage = keep_usage

    restores, restore = [], gemini_client.restore_original_face

    def keep_restore(*a, **k):
        result = restore(*a, **k)
        restores.append("restored" if result[1].restored else ("skipped: " + result[1].reason[:40]))
        return result
    gemini_client.restore_original_face = keep_restore

    capture, dress = data_url(capture_path), data_url(garment_path)
    results = []
    stdout = sys.stdout
    try:
        with TestClient(app_main.app) as client:
            tryon.warm_up_yolo()
            profile = client.post("/api/extract-profile", json={"image": capture, "hold_frames": [], "session_id": "sess_bench_profile"}).json()
            print("Preparing the capture (as when a visitor accepts the photo)...", flush=True)
            client.post("/api/prepare-capture", json={"image": capture})
            deadline = time.time() + 60
            while time.time() < deadline and not gemini_client._USER_APPEARANCE_CACHE:
                time.sleep(0.5)
            for round_no in range(1, args.runs + 1):
                for model, size in configs:
                    gemini_client.TRYON_MODEL, gemini_client.TRYON_IMAGE_SIZE = model, size
                    label = f"{model}_{size}_{round_no}"
                    print(f"{label}...", flush=True)
                    before = (len(timing_rows), len(usage), len(restores))
                    payload = {"user_image": capture, "dress_image": dress, "selected_size": None,
                               "body_profile": profile.get("body_profile"),
                               "session_id": f"sess_bench_{int(time.time())}_{round_no}"}
                    sys.stdout = open(os.devnull, "w", encoding="utf-8")
                    try:
                        resp = client.post("/api/tryon", json=payload)
                    finally:
                        sys.stdout.close()
                        sys.stdout = stdout
                    body = resp.json() if resp.status_code == 200 else {}
                    row = timing_rows[before[0]] if len(timing_rows) > before[0] else {}
                    used = usage[before[1]] if len(usage) > before[1] else {}
                    if body.get("result_image"):
                        encoded = body["result_image"].split(",", 1)[1]
                        with open(os.path.join(out_dir, f"{label}.jpg"), "wb") as fh:
                            fh.write(base64.b64decode(encoded))
                    results.append({
                        "model": model, "size": size, "round": round_no, "http": resp.status_code,
                        "result": row.get("attempt_results") or row.get("outcome", ""),
                        "total_s": row.get("total_s", ""), "gemini_s": row.get("gemini_image_s", ""),
                        "local_s": f"{float(row['total_s']) - float(row['gemini_image_s']):.2f}" if row else "",
                        "image_tokens": used.get("tokens", ""), "cost_usd": used.get("cost") or "",
                        "silhouette": body.get("silhouette_score", ""), "face_similarity": body.get("face_similarity", ""),
                        "coverage_ok": body.get("coverage_verified", ""),
                        "face_restore": restores[before[2]] if len(restores) > before[2] else "",
                    })
                    print(f"  HTTP {resp.status_code} {results[-1]['result']} | total {results[-1]['total_s']}s | "
                          f"gemini {results[-1]['gemini_s']}s | {results[-1]['face_restore']}", flush=True)
                    if resp.status_code != 200:
                        print("  " + resp.text[:200], flush=True)
    finally:
        sys.stdout = stdout
        if results:
            with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)

    print(f"\n{'config':<34}{'runs':>5}{'total p50':>11}{'gemini p50':>12}{'local p50':>11}{'checks ok':>11}")
    for model, size in configs:
        rows = [r for r in results if (r["model"], r["size"]) == (model, size) and r["total_s"] and r["http"] == 200]
        if not rows:
            print(f"{model + ' ' + size:<34}{0:>5}   no successful run")
            continue
        median = lambda key: statistics.median(float(r[key]) for r in rows)
        passed = sum(r["result"] == "ok" for r in rows)
        print(f"{model + ' ' + size:<34}{len(rows):>5}{median('total_s'):>11.1f}{median('gemini_s'):>12.1f}"
              f"{median('local_s'):>11.1f}{f'{passed}/{len(rows)}':>11}")
    print(f"\nOutputs and results.csv: {out_dir}")


if __name__ == "__main__":
    main()
