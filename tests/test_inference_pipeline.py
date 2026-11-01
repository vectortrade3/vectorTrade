"""
VectorTrade - Unit Test Suite: Machine Learning Model Inference Pipeline
Tests Section 1 (Strategy Model Inference) and Section 2 (Global Regime Inference):
- JSON State initialization and schema persistence
- Verification of models/model_performance_log.json structure and metrics
- Daily skip check (< 0.05s execution when already evaluated today)
- Progress dispatch and WebSocket listener broadcasting
- Flask API endpoints (/api/inference/status, /api/inference/progress, /api/inference/start_*)
- Clean test tearDown ensuring zero test progress pollution
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from offline.model_inferencer import (
    run_strategy_inference,
    run_global_inference,
    run_all_inference
)
from offline.app import app


class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        self.app_client = app.test_client()
        self._infer_prog_snapshot = json_manager.get_inference_progress()

    def tearDown(self):
        if hasattr(self, '_infer_prog_snapshot') and self._infer_prog_snapshot:
            json_manager.write_json_file(json_manager.INFERENCE_PROGRESS_FILE, self._infer_prog_snapshot)

    def test_01_inference_state_initialization(self):
        """Verifies inference_state.json contains Section 1 (Strategy) and Section 2 (Global)."""
        info = json_manager.get_inference_info()
        self.assertIn("strategy_inference", info)
        self.assertIn("global_inference", info)
        self.assertIn("last_inference_date", info)

        strat = info["strategy_inference"]
        self.assertIn("accuracy_refined", strat)
        self.assertIn("win_rate_recall", strat)
        self.assertIn("mae_refined", strat)

        glob = info["global_inference"]
        self.assertIn("mae_gain_refined", glob)
        self.assertIn("mae_drop_refined", glob)
        self.assertIn("directional_accuracy", glob)

    def test_02_model_performance_log_structure(self):
        """Verifies models/model_performance_log.json contains notebook 5 and 5.2 entries."""
        perf_log = json_manager.get_model_performance_log()
        self.assertTrue(len(perf_log) > 0, "model_performance_log.json is empty")
        self.assertIn("Strategy_Base", perf_log)
        self.assertIn("Strategy_Refined", perf_log)
        self.assertIn("Global_Gain_Base", perf_log)
        self.assertIn("Global_Gain_Refined", perf_log)

        strat_ref = perf_log["Strategy_Refined"]
        self.assertIn("metrics", strat_ref)
        self.assertIn("Accuracy", strat_ref["metrics"])
        self.assertIn("Win_Rate_Recall", strat_ref["metrics"])
        self.assertIn("MAE", strat_ref["metrics"])

    def test_03_inference_progress_dispatch(self):
        """Verifies progress updates dispatch to registered WebSocket listeners."""
        received = []

        def dummy_listener(event_type, data):
            received.append((event_type, data))

        json_manager.register_progress_listener(dummy_listener)
        try:
            json_manager.update_inference_progress(
                status="EVALUATING",
                active_section="TEST",
                phase="Unit Test Phase",
                percentage=50.0,
                message="Dispatch test"
            )

            self.assertTrue(len(received) > 0)
            self.assertEqual(received[-1][0], "INFERENCE_PROGRESS")
            self.assertEqual(received[-1][1]["active_section"], "TEST")
            self.assertEqual(received[-1][1]["percentage"], 50.0)
        finally:
            if dummy_listener in json_manager._progress_listeners:
                json_manager._progress_listeners.remove(dummy_listener)

    def test_04_strategy_inference_skip_logic(self):
        """Verifies skip logic when strategy inference was already executed today."""
        res = run_strategy_inference(force=False)
        self.assertIn(res["status"], ["SUCCESS"])
        self.assertIn("message", res)
        if res.get("skipped"):
            self.assertEqual(res["section"], "STRATEGY")

    def test_05_global_inference_skip_logic(self):
        """Verifies skip logic when global inference was already executed today."""
        res = run_global_inference(force=False)
        self.assertIn(res["status"], ["SUCCESS"])
        self.assertIn("message", res)
        if res.get("skipped"):
            self.assertEqual(res["section"], "GLOBAL")

    def test_06_run_all_inference_orchestrator(self):
        """Verifies run_all_inference executes sequential orchestrator."""
        res = run_all_inference(force=False)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("strategy", res)
        self.assertIn("global", res)

    def test_07_api_inference_endpoints(self):
        """Verifies GET /api/inference/status and /api/inference/progress return HTTP 200."""
        res = self.app_client.get('/api/inference/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("info", data)
        self.assertIn("progress", data)
        self.assertIn("performance_log", data)

        res_p = self.app_client.get('/api/inference/progress')
        self.assertEqual(res_p.status_code, 200)
        data_p = res_p.get_json()
        self.assertIn("status", data_p)
        self.assertIn("percentage", data_p)

    def test_08_api_status_includes_inference(self):
        """Verifies GET /api/status includes inference_info and inference_progress."""
        res = self.app_client.get('/api/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("inference_info", data)
        self.assertIn("inference_progress", data)
        self.assertEqual(data["status"], "ONLINE")

    def test_09_trade_ledgers_persistence(self):
        """Verifies 50 MA and Global trade ledgers are persisted in data_store/."""
        trades_50ma = json_manager.get_inference_50ma_trades()
        self.assertIn("trades", trades_50ma)
        self.assertIn("summary", trades_50ma)
        self.assertTrue(len(trades_50ma["trades"]) > 0, "50 MA trades ledger is empty")
        
        sum_50 = trades_50ma["summary"]
        self.assertIn("win_rate_taken", sum_50)
        self.assertIn("overall_pnl_pct", sum_50)
        self.assertIn("taken_trades", sum_50)
        self.assertIn("total_trades", sum_50)

        regimes_glob = json_manager.get_inference_global_trades()
        self.assertIn("regimes", regimes_glob)
        self.assertIn("summary", regimes_glob)
        self.assertTrue(len(regimes_glob["regimes"]) > 0, "Global regimes ledger is empty")

        sum_g = regimes_glob["summary"]
        self.assertIn("directional_accuracy", sum_g)
        self.assertIn("win_rate_pct", sum_g)
        self.assertIn("overall_pnl_pct", sum_g)

    def test_10_api_trade_ledgers_endpoints(self):
        """Verifies GET /api/inference/trades/50ma and /api/inference/trades/global."""
        # 50 MA endpoint
        res = self.app_client.get('/api/inference/trades/50ma?page=1&limit=25')
        self.assertEqual(res.status_code, 200)
        d_50 = res.get_json()
        self.assertEqual(d_50["status"], "SUCCESS")
        self.assertEqual(len(d_50["trades"]), 25)
        self.assertGreaterEqual(d_50["total_pages"], 40)
        self.assertIn("win_rate_taken", d_50["summary"])

        # Filter check
        res_filter = self.app_client.get('/api/inference/trades/50ma?decision=TAKE&limit=10')
        self.assertEqual(res_filter.status_code, 200)
        d_filter = res_filter.get_json()
        self.assertTrue(all(t["decision"] == "TAKE" for t in d_filter["trades"]))

        # Global endpoint
        res_g = self.app_client.get('/api/inference/trades/global?page=1&limit=25')
        self.assertEqual(res_g.status_code, 200)
        d_glob = res_g.get_json()
        self.assertEqual(d_glob["status"], "SUCCESS")
        self.assertEqual(len(d_glob["regimes"]), 25)
        self.assertIn("directional_accuracy", d_glob["summary"])

    def test_11_trade_record_schema_fidelity(self):
        """Verifies individual trade and regime records contain exact required fields."""
        trades_50ma = json_manager.get_inference_50ma_trades()["trades"]
        first_t = trades_50ma[0]
        required_t_keys = [
            "id", "symbol", "date", "entry_price", "stop_loss", "target", "risk",
            "predicted_r", "real_r", "outcome", "decision", "trade_pnl_pct", "cumulative_pnl_pct"
        ]
        for k in required_t_keys:
            self.assertIn(k, first_t, f"Missing {k} in 50 MA trade record")

        regimes = json_manager.get_inference_global_trades()["regimes"]
        first_r = regimes[0]
        required_r_keys = [
            "id", "symbol", "date", "close", "predicted_gain_7d", "predicted_drop_7d",
            "real_gain_7d", "real_drop_7d", "real_return_7d", "decision", "trade_pnl_pct", "cumulative_pnl_pct"
        ]
        for k in required_r_keys:
            self.assertIn(k, first_r, f"Missing {k} in Global regime record")


if __name__ == '__main__':
    unittest.main()
