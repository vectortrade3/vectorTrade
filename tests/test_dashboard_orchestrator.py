import sys
from datetime import datetime
import json
import pytest

sys.path.insert(0, '.')
from offline.app import app
from offline import json_manager

class TestDashboardOrchestrator:

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = app.test_client()

    def test_01_api_status_comprehensive_payload(self):
        res = self.client.get('/api/status')
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("status") == "ONLINE"
        
        # Verify all 5 stages metadata are present
        assert "download_info" in data
        assert "feature_gen_info" in data
        assert "data_cleaning_info" in data
        assert "models_info" in data
        assert "inference_info" in data

        # Check date fields
        assert "last_downloaded_date" in data["download_info"]
        assert "last_feature_gen_date" in data["feature_gen_info"]
        assert "last_clean_date" in data["data_cleaning_info"]

    def test_02_all_stage_progress_endpoints(self):
        endpoints = [
            '/api/download/progress',
            '/api/feature_gen/progress',
            '/api/data_clean/progress',
            '/api/training/progress',
            '/api/inference/progress'
        ]
        for ep in endpoints:
            res = self.client.get(ep)
            assert res.status_code == 200, f"Failed on endpoint {ep}"
            prog = res.get_json()
            assert "status" in prog, f"Missing status in {ep}"
            assert prog["status"] in ("IDLE", "CHECKING", "DOWNLOADING", "APPENDING",
                                      "GENERATING", "SAVING", "COMBINING", "CLEANING",
                                      "TRAINING", "INITIALIZING", "EVALUATING", "COMPLETED", "ERROR")

    def test_03_done_for_today_persistence_consistency(self):
        today_str = datetime.now().astimezone().strftime("%Y-%m-%d")
        
        # Verify json_manager methods can retrieve or set run date
        down_info = json_manager.get_stocks_download_info()
        assert "last_download_run_date" in down_info
        
        feat_info = json_manager.get_feature_gen_info()
        assert "last_feature_run_date" in feat_info
        
        clean_info = json_manager.get_data_cleaning_info()
        assert "last_clean_run_date" in clean_info

    def test_04_html_template_renders_with_mocked_dates(self):
        today_str = datetime.now().astimezone().strftime("%Y-%m-%d")
        res = self.client.get('/')
        assert res.status_code == 200
        html = res.data.decode('utf-8')
        
        # Verify 5 cards with all required IDs exist in the rendered HTML
        for card_id in ["dashCard-download", "dashCard-features", "dashCard-cleaning", "dashCard-training", "dashCard-inference"]:
            assert f'id="{card_id}"' in html, f"Missing {card_id}"

        for badge_id in ["dashBadge-download", "dashBadge-features", "dashBadge-cleaning", "dashBadge-training", "dashBadge-inference"]:
            assert f'id="{badge_id}"' in html, f"Missing {badge_id}"

        for date_id in ["dashDate-download", "dashDate-features", "dashDate-cleaning", "dashDate-training", "dashDate-inference"]:
            assert f'id="{date_id}"' in html, f"Missing {date_id}"

        for today_id in ["dashToday-download", "dashToday-features", "dashToday-cleaning"]:
            assert f'id="{today_id}"' in html, f"Missing {today_id}"

    def test_05_orchestrator_js_syntax_integrity(self):
        res = self.client.get('/')
        assert res.status_code == 200
        html = res.data.decode('utf-8')

        # Check that error halt logic is present in the orchestrator
        assert "handleStageError" in html
        assert "setPipelineStageCardState(stages[j], 'idle')" in html
        assert "pipelineAbortRequested" in html
        assert "checkAborted" in html
        assert "isStageDoneForToday" in html
        assert "already completed today. Skipping to next stage" in html

    def test_06_model_trainer_concurrency_safety(self):
        # Verify model_trainer avoids n_jobs=-1 in RandomizedSearchCV to prevent Windows process termination
        with open('offline/model_trainer.py', 'r', encoding='utf-8') as f:
            code = f.read()
        assert "RandomizedSearchCV(" in code
        assert "n_jobs=-1" not in code, "Found dangerous n_jobs=-1 in model_trainer.py"
        assert "n_jobs=1" in code, "Expected n_jobs=1 in RandomizedSearchCV"

    def test_07_feature_generator_flushing_progress(self):
        # Verify feature_generator emits incremental saving progress
        with open('offline/feature_generator.py', 'r', encoding='utf-8') as f:
            code = f.read()
        assert "Flushing engineered CSV" in code
        assert "total_saved = len(generated_results)" in code

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
