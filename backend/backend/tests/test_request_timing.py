import csv
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services import gemini_cost, request_timing
from app.services.csv_log import append_row


class TestRequestTiming(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "logs", "tryon_timings.csv")
        for p in (patch.object(request_timing, "TRYON_TIMING_LOG", True),
                  patch.object(request_timing, "TRYON_TIMING_LOG_PATH", self.path)):
            p.start()
            self.addCleanup(p.stop)

    def test_lap_leaves_out_step_blocks_so_nothing_is_counted_twice(self):
        timing = request_timing.begin("sess_abcdefgh")
        time.sleep(0.02)
        with request_timing.step("gemini_image"):
            time.sleep(0.05)
        request_timing.lap("checks_after")
        self.assertAlmostEqual(timing.steps["gemini_image"], 0.05, delta=0.02)
        self.assertAlmostEqual(timing.steps["checks_after"], 0.02, delta=0.015)
        row = request_timing.finish(timing, "served", "L", "perfect")
        parts = sum(float(row[f"{name}_s"]) for name in request_timing.STEPS) + float(row["other_s"])
        self.assertAlmostEqual(parts, float(row["total_s"]), delta=0.02)

    def test_one_row_per_request_with_attempt_results(self):
        timing = request_timing.begin("sess_abcdefgh")
        request_timing.attempt_result("silhouette")
        request_timing.attempt_result("ok")
        request_timing.finish(timing, "served", "XL", "loose")
        request_timing.finish(request_timing.begin(None), "cache_hit", "M", None)
        with open(self.path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["session"], rows[0]["attempts"], rows[0]["attempt_results"]), ("sess_abc", "2", "silhouette;ok"))
        self.assertEqual((rows[1]["outcome"], rows[1]["attempts"]), ("cache_hit", "0"))

    def test_no_op_outside_a_timed_request(self):
        request_timing._current.set(None)
        with request_timing.step("finalize"):
            pass
        request_timing.lap("checks_before")
        request_timing.attempt_result("ok")
        self.assertFalse(os.path.exists(self.path))


class TestCsvLog(unittest.TestCase):
    def test_older_column_set_is_kept_aside(self):
        path = os.path.join(tempfile.mkdtemp(), "logs", "x.csv")
        append_row(path, ["a", "b"], {"a": 1, "b": 2})
        append_row(path, ["a", "b", "c"], {"a": 3, "b": 4, "c": 5})
        folder = os.path.dirname(path)
        kept = [f for f in os.listdir(folder) if f.startswith("x_until_")]
        self.assertEqual(len(kept), 1)
        with open(path, newline="", encoding="utf-8") as fh:
            self.assertEqual(list(csv.DictReader(fh)), [{"a": "3", "b": "4", "c": "5"}])


class TestGeminiCostDuration(unittest.TestCase):
    def test_duration_column(self):
        path = os.path.join(tempfile.mkdtemp(), "logs", "gemini_costs.csv")
        usage = SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=2520, thoughts_token_count=0, candidates_tokens_details=None))
        with patch.object(gemini_cost, "GEMINI_COST_LOG", True), patch.object(gemini_cost, "GEMINI_COST_LOG_PATH", path):
            gemini_cost.record_usage("tryon_generation", gemini_cost.FLASH_IMAGE_MODEL, usage, duration_s=36.214)
            gemini_cost.record_usage("cloth_analysis", gemini_cost.ANALYSIS_MODEL, usage)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["duration_s"] for r in rows], ["36.21", ""])


if __name__ == "__main__":
    unittest.main()
