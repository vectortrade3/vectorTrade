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

    # Control buttons checks
    assert 'btnDashDownload' in html, 'btnDashDownload missing'
    assert 'btnDashFeatureGen' in html, 'btnDashFeatureGen missing'
    assert 'btnDashDataClean' in html, 'btnDashDataClean missing'
    assert 'btnDashTrainAll' in html, 'btnDashTrainAll missing'
    assert 'btnDashInferAll' in html, 'btnDashInferAll missing'

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

    # Default tab ID check
    assert "let activeTabId = 'dashboard';" in html, 'activeTabId should be dashboard'

    print("SUCCESS: All Dashboard DOM and template checks passed!")

if __name__ == '__main__':
    test_dashboard_template()
