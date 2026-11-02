import sys
sys.path.insert(0, '.')
from offline.app import app

def test_dashboard_template():
    client = app.test_client()
    res = client.get('/')
    assert res.status_code == 200, f'Status {res.status_code}'
    html = res.data.decode('utf-8')

    # View checks
    assert 'id="tab-dashboard-view"' in html, 'tab-dashboard-view missing'
    assert 'id="tab-download-view" class="hidden space-y-6"' in html, 'tab-download-view should be hidden initially'
    assert 'id="nav-dashboard"' in html, 'nav-dashboard missing'

    # Master Hero Section & Orchestrator CTA
    assert 'id="btnRunAllAnalysis"' in html, 'btnRunAllAnalysis missing in Hero'
    assert 'onclick="triggerRunAllAnalysis()"' in html, 'triggerRunAllAnalysis onclick missing'
    assert 'id="btnAbortPipeline"' in html, 'btnAbortPipeline missing'
    assert 'id="pipelineStatusDot"' in html, 'pipelineStatusDot missing'
    assert 'id="pipelineStatusText"' in html, 'pipelineStatusText missing'

    # Individual control buttons checks
    assert 'btnDashDownload' in html, 'btnDashDownload missing'
    assert 'btnDashFeatureGen' in html, 'btnDashFeatureGen missing'
    assert 'btnDashDataClean' in html, 'btnDashDataClean missing'
    assert 'btnDashTrainAll' in html, 'btnDashTrainAll missing'
    assert 'btnDashInferAll' in html, 'btnDashInferAll missing'

    # 5 Card Containers
    assert 'id="dashCard-download"' in html, 'dashCard-download missing'
    assert 'id="dashCard-features"' in html, 'dashCard-features missing'
    assert 'id="dashCard-cleaning"' in html, 'dashCard-cleaning missing'
    assert 'id="dashCard-training"' in html, 'dashCard-training missing'
    assert 'id="dashCard-inference"' in html, 'dashCard-inference missing'

    # 5 Card State Badges (Default IDLE)
    assert 'id="dashBadge-download"' in html, 'dashBadge-download missing'
    assert 'id="dashBadge-features"' in html, 'dashBadge-features missing'
    assert 'id="dashBadge-cleaning"' in html, 'dashBadge-cleaning missing'
    assert 'id="dashBadge-training"' in html, 'dashBadge-training missing'
    assert 'id="dashBadge-inference"' in html, 'dashBadge-inference missing'
    assert '>IDLE</span>' in html, 'Default IDLE badge text missing'

    # Date and "Done for today" pills
    assert 'id="dashDate-download"' in html, 'dashDate-download missing'
    assert 'id="dashDate-features"' in html, 'dashDate-features missing'
    assert 'id="dashDate-cleaning"' in html, 'dashDate-cleaning missing'
    assert 'id="dashDate-training"' in html, 'dashDate-training missing'
    assert 'id="dashDate-inference"' in html, 'dashDate-inference missing'

    assert 'id="dashToday-download"' in html, 'dashToday-download missing'
    assert 'id="dashToday-features"' in html, 'dashToday-features missing'
    assert 'id="dashToday-cleaning"' in html, 'dashToday-cleaning missing'
    assert 'id="dashToday-training"' in html, 'dashToday-training missing'
    assert 'id="dashToday-inference"' in html, 'dashToday-inference missing'
    assert 'Done for today' in html, 'Done for today text missing'

    # Live progress bar containers
    assert 'dashProgressDownload' in html, 'dashProgressDownload missing'
    assert 'dashProgressFeatures' in html, 'dashProgressFeatures missing'
    assert 'dashProgressCleaning' in html, 'dashProgressCleaning missing'
    assert 'dashProgressTraining' in html, 'dashProgressTraining missing'
    assert 'dashProgressInference' in html, 'dashProgressInference missing'

    # Post-completion inspection links for steps 4 and 5
    assert 'dashTrainingResultLink' in html, 'dashTrainingResultLink missing'
    assert 'dashInferenceResultLink' in html, 'dashInferenceResultLink missing'
    assert "switchTab('training')" in html, 'switchTab(training) missing'
    assert "switchTab('inference')" in html, 'switchTab(inference) missing'

    # JavaScript Orchestrator functions
    assert 'function triggerRunAllAnalysis()' in html, 'triggerRunAllAnalysis missing in JS'
    assert 'function setPipelineStageCardState(' in html, 'setPipelineStageCardState missing in JS'
    assert 'function checkAndApplyDoneForTodayPills(' in html, 'checkAndApplyDoneForTodayPills missing in JS'
    assert 'function executeStageAndWait(' in html, 'executeStageAndWait missing in JS'
    assert 'function abortPipeline()' in html, 'abortPipeline missing in JS'
    assert 'const STAGE_CONFIG =' in html, 'STAGE_CONFIG map missing in JS'

    # Default tab ID check
    assert "let activeTabId = 'dashboard';" in html, 'activeTabId should be dashboard'

    print("SUCCESS: All Dashboard DOM and template checks passed!")

if __name__ == '__main__':
    test_dashboard_template()
