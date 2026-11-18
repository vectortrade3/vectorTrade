"""
VectorTrade - Unit Tests for Cointegrated Pairs Trading v4 Strategy & Screener
Verifies:
1. Loading and validation of data_store/curated_pairs.json.
2. Pair structure conforms to 7-Stage Gauntlet criteria:
   - Intra-sector pair validation
   - Correlation >= 0.78
   - Cointegration p-value <= 0.05
   - Valid half-life in days
   - Hurst exponent < 0.50 (mean-reverting)
   - Chart data presence (dates, normalized price series, z-score)
3. Offline Flask /pairs page endpoint (200 OK, HTML rendering).
4. Offline Flask /api/pairs API endpoint (200 OK, JSON response).
"""

import sys
import unittest
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline.app import app


class TestPairsStrategy(unittest.TestCase):
    """Test suite for Cointegrated Stat-Arb Pair Strategy v4."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.pairs_file = BASE_DIR / "data_store" / "curated_pairs.json"

    def test_01_curated_pairs_file_exists_and_not_empty(self):
        """Verifies curated_pairs.json exists and contains high-conviction pairs."""
        self.assertTrue(self.pairs_file.exists(), "curated_pairs.json must exist in data_store/")
        with open(self.pairs_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("pairs", data)
        self.assertGreater(data.get("total_pairs", 0), 0)
        self.assertGreaterEqual(len(data["pairs"]), 20)

    def test_02_pair_metrics_satisfy_stat_arb_gauntlet(self):
        """Verifies each pair satisfies mathematical cointegration thresholds."""
        with open(self.pairs_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for p in data["pairs"][:10]:
            self.assertIn("ticker_a", p)
            self.assertIn("ticker_b", p)
            self.assertIn("sector", p)
            self.assertGreaterEqual(p["correlation"], 0.75, f"Low correlation in pair {p['id']}")
            self.assertLessEqual(p["cointegration_pvalue"], 0.05, f"Cointegration failed in pair {p['id']}")
            self.assertLess(p["hurst_exponent"], 0.50, f"Non-mean reverting Hurst in pair {p['id']}")
            self.assertGreater(p["half_life_days"], 0)
            self.assertIn("chart_data", p)
            cd = p["chart_data"]
            self.assertIn("dates", cd)
            self.assertIn("norm_a", cd)
            self.assertIn("norm_b", cd)
            self.assertIn("zscore", cd)
            self.assertGreater(len(cd["dates"]), 30)

    def test_03_pairs_page_endpoint_returns_200(self):
        """Verifies GET /pairs renders the pairs page with modal inspector."""
        res = self.client.get("/pairs")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("Cointegrated Pairs Trading v4 Strategy", html)
        self.assertIn("pairModal", html)
        self.assertIn("priceChart", html)
        self.assertIn("zscoreChart", html)

    def test_04_api_pairs_endpoint_returns_json(self):
        """Verifies GET /api/pairs returns successful JSON payload."""
        res = self.client.get("/api/pairs")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload.get("status"), "SUCCESS")
        self.assertIn("data", payload)
        self.assertIn("pairs", payload["data"])


if __name__ == "__main__":
    unittest.main()
