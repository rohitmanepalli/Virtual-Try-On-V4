import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services import gemini_cost
from app.services.gemini_cost import ANALYSIS_MODEL, FLASH_IMAGE_MODEL, PRO_IMAGE_MODEL, TRYON_MODEL, record_usage


def _usage(prompt, out, thoughts=0, details=None):
    return SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=prompt, candidates_token_count=out, thoughts_token_count=thoughts,
        candidates_tokens_details=details))


def _detail(modality, count):
    return SimpleNamespace(modality=SimpleNamespace(name=modality), token_count=count)


class TestGeminiCost(unittest.TestCase):
    def setUp(self):
        p = patch.object(gemini_cost, "GEMINI_COST_LOG", False)
        p.start()
        self.addCleanup(p.stop)

    def test_4k_tryon_image_priced_from_modality_breakdown(self):
        # 3 input images + prompt, one 4K image (2520 tokens) and a short text part
        resp = _usage(4000, 2560, thoughts=300, details=[_detail("IMAGE", 2520), _detail("TEXT", 40)])
        cost = record_usage("tryon_generation", FLASH_IMAGE_MODEL, resp, session_id="abcdef123456", size="L", attempt=1)
        expected = (4000 * 0.50 + (40 + 300) * 3.00 + 2520 * 60.00) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=9)
        self.assertAlmostEqual(2520 * 60.00 / 1_000_000, 0.1512, places=4)  # Google's published ~$0.151 per 4K image

    def test_pro_4k_image_priced_at_pro_rates(self):
        resp = _usage(600, 2150, thoughts=200, details=[_detail("IMAGE", 2000), _detail("TEXT", 150)])
        cost = record_usage("tryon_generation", PRO_IMAGE_MODEL, resp)
        expected = (600 * 2.00 + (150 + 200) * 12.00 + 2000 * 120.00) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=9)
        self.assertAlmostEqual(2000 * 120.00 / 1_000_000, 0.24, places=4)  # Google's published ~$0.24 per 4K image

    def test_image_model_without_breakdown_treats_output_as_image(self):
        cost = record_usage("tryon_generation", FLASH_IMAGE_MODEL, _usage(1000, 2520))
        self.assertAlmostEqual(cost, (1000 * 0.50 + 2520 * 60.00) / 1_000_000, places=9)

    def test_analysis_model_output_is_text(self):
        cost = record_usage("appearance_analysis", ANALYSIS_MODEL, _usage(1500, 400, thoughts=600))
        self.assertAlmostEqual(cost, (1500 * 0.30 + 1000 * 2.50) / 1_000_000, places=9)

    def test_missing_usage_or_unknown_model_is_not_priced(self):
        self.assertIsNone(record_usage("x", FLASH_IMAGE_MODEL, SimpleNamespace(usage_metadata=None)))
        self.assertIsNone(record_usage("x", "some-other-model", _usage(10, 10)))

    def test_request_cost_tracking_sums_this_requests_calls(self):
        costs = gemini_cost.begin_cost_tracking()
        record_usage("tryon_generation", FLASH_IMAGE_MODEL, _usage(1000, 2520), size="L", attempt=1)
        record_usage("tryon_generation", FLASH_IMAGE_MODEL, _usage(1000, 2520), size="L", attempt=2)
        self.assertEqual(len(costs), 2)
        self.assertAlmostEqual(sum(costs), 2 * (1000 * 0.50 + 2520 * 60.00) / 1_000_000, places=9)


class TestCostLogFile(unittest.TestCase):
    def setUp(self):
        import tempfile, os
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "gemini_costs.csv")
        for p in (patch.object(gemini_cost, "GEMINI_COST_LOG", True), patch.object(gemini_cost, "GEMINI_COST_LOG_PATH", self.path)):
            p.start()
            self.addCleanup(p.stop)

    def _rows(self):
        import csv
        with open(self.path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def test_row_carries_fit_category_and_offset(self):
        record_usage("tryon_generation", TRYON_MODEL, _usage(1000, 2520), session_id="sess_abcdefgh", size="XXL",
                     attempt=1, fit_category="oversized", size_offset=3)
        record_usage("tryon_generation", TRYON_MODEL, _usage(1000, 2520), size="XS", fit_category="does_not_fit", size_offset=-2)
        rows = self._rows()
        self.assertEqual([(r["size"], r["fit_category"], r["size_offset"]) for r in rows],
                         [("XXL", "oversized", "3"), ("XS", "does_not_fit", "-2")])
        self.assertTrue(rows[0]["cost_usd"])

    def test_old_format_log_is_kept_aside_not_corrupted(self):
        import glob, os
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("timestamp,kind,model,session,size,attempt,input_tokens,text_out_tokens,image_out_tokens,thinking_tokens,cost_usd\n")
            fh.write("2026-09-28T11:07:07,tryon_generation,m,s,M,1,1,1,1,0,0.15\n")
        record_usage("tryon_generation", TRYON_MODEL, _usage(1000, 2520), size="L", fit_category="relaxed", size_offset=1)
        self.assertEqual([r["fit_category"] for r in self._rows()], ["relaxed"])
        kept = glob.glob(os.path.join(self.dir, "gemini_costs_until_*.csv"))
        self.assertEqual(len(kept), 1)
        with open(kept[0], encoding="utf-8") as fh:
            self.assertIn("0.15", fh.read())


if __name__ == "__main__":
    unittest.main()
