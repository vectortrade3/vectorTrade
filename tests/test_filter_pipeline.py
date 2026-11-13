"""
VectorTrade - Unit Tests for Interactive Filter Workflow Pipeline
Tests:
1. Retrieval of default pipeline stages with empirical scorecard metrics.
2. Preservation of fixed start (primary_model) and fixed end (final_execution_list).
3. Dynamic shifting / re-ordering of intermediate stages.
4. Enable/disable toggle (bypass logic) for intermediate stages.
5. Dynamic priority sorting with custom pipeline configurations.
6. REST API endpoints:
   - GET /api/trading-settings/filter-pipeline
   - POST /api/trading-settings/filter-pipeline
   - POST /api/trading-settings/filter-pipeline/reset
"""

import unittest
from unittest.mock import patch, MagicMock
from online.services.filter_pipeline_service import FilterPipelineService, get_default_pipeline_stages
from offline.app import app as flask_app


class TestFilterPipeline(unittest.TestCase):
    """Test suite for interactive filter workflow pipeline service and APIs."""

    def setUp(self):
        self.service = FilterPipelineService()
        self.client = flask_app.test_client()

    def test_01_default_pipeline_structure_and_metrics(self):
        """Verifies default pipeline has fixed start, shiftable intermediate stages, and end rule."""
        stages = self.service.get_pipeline_configuration(force_refresh=True)
        self.assertGreaterEqual(len(stages), 5)

        # 1. Fixed Start
        self.assertEqual(stages[0]["id"], "primary_model")
        self.assertTrue(stages[0]["is_fixed"])
        self.assertFalse(stages[0]["can_shift"])
        self.assertTrue(stages[0]["has_model_results"])

        # 2. Fixed End
        self.assertEqual(stages[-1]["id"], "final_execution_list")
        self.assertTrue(stages[-1]["is_fixed"])
        self.assertFalse(stages[-1]["can_shift"])

        # 3. Intermediate Shiftable Stages
        intermediate = [s for s in stages if s.get("can_shift")]
        intermediate_ids = [s["id"] for s in intermediate]
        self.assertIn("multi_strategy_confluence", intermediate_ids)
        self.assertIn("secondary_stock_model", intermediate_ids)
        self.assertIn("quality_classifier", intermediate_ids)

        # 4. Scorecard Metrics
        quality_stage = next(s for s in stages if s["id"] == "quality_classifier")
        self.assertTrue(quality_stage["has_model_results"])
        self.assertIn("accuracy", quality_stage["metrics"])
        self.assertIn("baseline_win_rate", quality_stage["metrics"])

    def test_02_reordering_intermediate_stages_preserves_boundaries(self):
        """Verifies shifting intermediate stages does not corrupt fixed start or fixed end."""
        initial_stages = self.service.get_pipeline_configuration(force_refresh=True)
        
        # Reverse only intermediate stages
        reordered = [
            {"id": "quality_classifier", "enabled": True},
            {"id": "secondary_stock_model", "enabled": True},
            {"id": "multi_strategy_confluence", "enabled": True},
            {"id": "alphabetical_sort", "enabled": True}
        ]

        with patch("online.db.DatabaseManager.trading_settings", None):
            updated = self.service.update_pipeline_configuration(reordered)

            # Boundaries must remain fixed
            self.assertEqual(updated[0]["id"], "primary_model")
            self.assertEqual(updated[-1]["id"], "final_execution_list")

            # Intermediate sequence updated
            updated_ids = [s["id"] for s in updated]
            q_idx = updated_ids.index("quality_classifier")
            c_idx = updated_ids.index("multi_strategy_confluence")
            self.assertLess(q_idx, c_idx)

    def test_03_stage_disable_bypass_logic(self):
        """Verifies disabling a filter bypasses it during prioritization."""
        cands = [
            {"symbol": "ZEE", "status": "ARMED", "is_multi_strategy": False, "stock_temp_model_score": 2.5},
            {"symbol": "TATA_MULTI", "status": "ARMED", "is_multi_strategy": True, "stock_temp_model_score": 1.2},
        ]

        # 1. With multi_strategy enabled: TATA_MULTI beats ZEE despite lower score
        pipe_enabled = [
            {"id": "multi_strategy_confluence", "enabled": True},
            {"id": "secondary_stock_model", "enabled": True}
        ]
        res1 = self.service.apply_pipeline(cands, custom_pipeline=pipe_enabled)
        self.assertEqual(res1[0]["symbol"], "TATA_MULTI")

        # 2. With multi_strategy disabled: bypassed! ZEE beats TATA_MULTI by score
        pipe_disabled = [
            {"id": "multi_strategy_confluence", "enabled": False},
            {"id": "secondary_stock_model", "enabled": True}
        ]
        res2 = self.service.apply_pipeline(cands, custom_pipeline=pipe_disabled)
        self.assertEqual(res2[0]["symbol"], "ZEE")

    def test_04_api_filter_pipeline_endpoints(self):
        """Verifies GET, POST, and RESET endpoints on Flask offline API."""
        # 1. GET
        res_get = self.client.get("/api/trading-settings/filter-pipeline")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.get_json()
        self.assertEqual(data_get["status"], "SUCCESS")
        self.assertIsInstance(data_get["pipeline"], list)

        # 2. POST Update
        stages_payload = [
            {"id": "quality_classifier", "enabled": True, "selected_threshold": 0.45},
            {"id": "secondary_stock_model", "enabled": False},
            {"id": "multi_strategy_confluence", "enabled": True}
        ]
        with patch("online.services.risk_calculator.risk_calculator.evaluate_and_allocate_candidate_orders") as mock_alloc:
            mock_alloc.return_value = {"confirmed_orders_count": 2}
            res_post = self.client.post("/api/trading-settings/filter-pipeline", json={"stages": stages_payload})
            self.assertEqual(res_post.status_code, 200)
            data_post = res_post.get_json()
            self.assertEqual(data_post["status"], "SUCCESS")

        # 3. POST Reset
        with patch("online.services.risk_calculator.risk_calculator.evaluate_and_allocate_candidate_orders") as mock_alloc:
            mock_alloc.return_value = {"confirmed_orders_count": 3}
            res_reset = self.client.post("/api/trading-settings/filter-pipeline/reset")
            self.assertEqual(res_reset.status_code, 200)
            data_reset = res_reset.get_json()
            self.assertEqual(data_reset["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
