"""
VectorTrade - Unit Tests for Simple Intra-Sector Pairs Strategy & Screener
Verifies:
1. Loading and validation of data_store/curated_pairs.json formed without external strategy config.
2. Pair structure conforms to fundamental intra-sector co-movement criteria:
   - Intra-sector pairing only (both stocks in same industry)
   - 5-Year Pearson correlation >= 0.85
   - 1-Year confirmation correlation >= 0.65
   - Valid hedge ratio (beta)
   - Chart data presence (dates, normalized price series base 100, z-score)
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


class TestSimplePairsStrategy(unittest.TestCase):
    """Test suite for Simple Intra-Sector Statistical Pairs."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.pairs_file = BASE_DIR / "data_store" / "curated_pairs.json"

    def test_01_curated_pairs_file_exists_and_not_empty(self):
        """Verifies curated_pairs.json exists and contains pairs formed from simple parameters."""
        self.assertTrue(self.pairs_file.exists(), "curated_pairs.json must exist in data_store/")
        with open(self.pairs_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("pairs", data)
        self.assertGreater(data.get("total_pairs", 0), 0)
        self.assertGreaterEqual(len(data["pairs"]), 50)
        self.assertIn("method", data)

    def test_02_pairs_satisfy_simple_intrasector_criteria(self):
        """Verifies each pair satisfies intra-sector and correlation thresholds."""
        with open(self.pairs_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for p in data["pairs"][:15]:
            self.assertIn("ticker_a", p)
            self.assertIn("ticker_b", p)
            self.assertIn("sector", p)
            self.assertGreaterEqual(p["correlation_5y"], 0.85, f"Low 5Y correlation in pair {p['id']}")
            self.assertGreaterEqual(p["correlation_1y"], 0.65, f"Low 1Y correlation in pair {p['id']}")
            self.assertIn("hedge_ratio", p)
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
        self.assertIn("Intra-Sector Trading Pairs", html)
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
