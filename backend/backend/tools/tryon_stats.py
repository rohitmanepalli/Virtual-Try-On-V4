"""
Summarises real try-on performance from the logs, at no cost: latency (average, P50, P95), each stage, first-pass success,
retries and API cost per successful generation.

    venv\\Scripts\\python.exe tools\\tryon_stats.py
    venv\\Scripts\\python.exe tools\\tryon_stats.py --since 2026-11-05T15:00:00

Cost per success needs request ids, which only rows written since they were added have. Older files are included in the
latency figures, but the older cost rows are not.
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVED = ("served", "wont_fit")


def read_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def percentile(values, share):
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * len(ordered) + 0.5) - 1))]


def summarise(timings, costs):
    generations = [r for r in timings if int(r.get("attempts") or 0) > 0]
    served = [r for r in generations if r["outcome"] in SERVED and r.get("attempt_results")]
    report = {
        "requests": len(timings),
        "outcomes": dict(Counter(r["outcome"] for r in timings)),
        "generated": len(generations),
        "served": len(served),
    }
    if not served:
        return report

    def stats(column):
        values = [float(r[column]) for r in served if r.get(column)]
        return {"avg": sum(values) / len(values), "p50": percentile(values, 0.50), "p95": percentile(values, 0.95)} if values else None

    columns = [c for c in served[-1] if c.endswith("_s")]
    report["stages"] = {c[:-2]: stats(c) for c in columns}
    report["first_pass"] = sum(r["attempt_results"].split(";")[0] == "ok" for r in generations) / len(generations)
    report["avg_attempts"] = sum(int(r["attempts"]) for r in generations) / len(generations)
    report["failed_checks"] = dict(Counter(result for r in generations for result in r["attempt_results"].split(";") if result != "ok"))

    served_ids = {r["request_id"] for r in served if r.get("request_id")}
    linked = [c for c in costs if c.get("request_id") in served_ids and c.get("cost_usd")]
    if served_ids and linked:
        report["cost_per_success"] = sum(float(c["cost_usd"]) for c in linked) / len({c["request_id"] for c in linked})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="", help="only rows at or after this ISO timestamp")
    parser.add_argument("--logs", default=os.path.join(BACKEND, "logs"))
    args = parser.parse_args()

    timings = [r for r in read_rows(os.path.join(args.logs, "tryon_timings*.csv")) if r["timestamp"] >= args.since]
    costs = [r for r in read_rows(os.path.join(args.logs, "gemini_costs*.csv")) if r["timestamp"] >= args.since]
    report = summarise(timings, costs)

    print(f"Requests: {report['requests']}  {report['outcomes']}")
    if "stages" not in report:
        sys.exit("No served generations in this period.")
    print(f"Generations served: {report['served']} of {report['generated']}")
    print(f"First-pass success: {report['first_pass']:.0%}   Attempts per generation: {report['avg_attempts']:.2f}")
    print(f"Failed checks: {report['failed_checks'] or 'none'}")
    if "cost_per_success" in report:
        print(f"API cost per successful generation: ${report['cost_per_success']:.3f}")
    print(f"\n{'stage':<18}{'avg':>8}{'P50':>8}{'P95':>8}   (seconds)")
    for stage, values in report["stages"].items():
        if values and values["p95"] >= 0.05:
            print(f"{stage:<18}{values['avg']:>8.1f}{values['p50']:>8.1f}{values['p95']:>8.1f}")


if __name__ == "__main__":
    main()
