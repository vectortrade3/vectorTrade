"""
Unit and Integration Tests for 50MA Strategy Data Page & TradingView Visualizer UI (Phase 3)
Verifies:
1. Nav button 'nav-ma50' and direct tab switching
2. Dashboard and Cleaning tab action triggers for 50MA Strategy Data
3. Section 'tab-ma50-view' presence with executive KPI cards, filters, and trade ledger table
4. TradingView-style candlestick modal 'modal50MAChart' with canvas, crosshair, and price badges
5. Template rendering integrity and status code 200 on Flask GET /
"""

import unittest
from pathlib import Path
from offline.app import app


class Test50MAVisualizerUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.html_path = Path(__file__).parent.parent / "offline" / "templates" / "index.html"
        with open(cls.html_path, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

    def test_01_nav_button_exists(self):
        """Test 1: Verifies 50MA Strategy nav button is properly embedded in header."""
        self.assertIn('id="nav-ma50"', self.html_content)
        self.assertIn("switchTab('ma50')", self.html_content)
        self.assertIn("50MA Trades", self.html_content)

    def test_02_dashboard_and_cleaning_card_buttons_exist(self):
        """Test 2: Verifies 50MA strategy buttons exist in Dashboard Stage 3 and Data Cleaning Tab."""
        self.assertIn('id="btnDash50MATrades"', self.html_content)
        self.assertIn("View 50MA Strategy Data", self.html_content)

    def test_03_ma50_view_section_structure(self):
        """Test 3: Verifies tab-ma50-view section exists with KPI cards, subclassification, and table."""
        self.assertIn('id="tab-ma50-view"', self.html_content)
        self.assertIn('id="stat50MATotalTrades"', self.html_content)
        self.assertIn('id="stat50MAWinRate1Y"', self.html_content)
        self.assertIn('id="stat50MAWinRate5Y"', self.html_content)
        self.assertIn('id="stat50MAAvgDays"', self.html_content)
        self.assertIn('id="stat50MAAvgPnL"', self.html_content)
        self.assertIn('id="table50MABody"', self.html_content)

    def test_04_split_and_outcome_filter_controls(self):
        """Test 4: Verifies split (TRAIN, TEST_1Y, MODEL_TAKE) and outcome filter pills exist."""
        self.assertIn('id="pill50MASplit-ALL"', self.html_content)
        self.assertIn('id="pill50MASplit-TRAIN"', self.html_content)
        self.assertIn('id="pill50MASplit-TEST_1Y"', self.html_content)
        self.assertIn('id="pill50MASplit-MODEL_TAKE"', self.html_content)
        self.assertIn('id="pill50MAOutcome-ALL"', self.html_content)
        self.assertIn('id="pill50MAOutcome-SUCCESS"', self.html_content)
        self.assertIn('id="pill50MAOutcome-LOSS"', self.html_content)
        self.assertIn('id="pill50MAOutcome-NEUTRAL"', self.html_content)

    def test_05_candlestick_modal_and_canvas_elements(self):
        """Test 5: Verifies custom TradingView-style candlestick chart modal and canvas exist."""
        self.assertIn('id="modal50MAChart"', self.html_content)
        self.assertIn('id="canvas50MAChart"', self.html_content)
        self.assertIn('id="modalChartSymbol"', self.html_content)
        self.assertIn('id="modalChartSector"', self.html_content)
        self.assertIn('id="modalChartOutcome"', self.html_content)
        self.assertIn('id="modalCoordEntry"', self.html_content)
        self.assertIn('id="modalCoordSL"', self.html_content)
        self.assertIn('id="modalCoordTarget"', self.html_content)
        self.assertIn('id="modalChartInspector"', self.html_content)

    def test_06_javascript_engine_functions_exist(self):
        """Test 6: Verifies required JavaScript functions are implemented in the client script."""
        self.assertIn("function load50MASummary()", self.html_content)
        self.assertIn("function load50MATrades()", self.html_content)
        self.assertIn("function open50MAChartModal(tradeId)", self.html_content)
        self.assertIn("function close50MAChartModal()", self.html_content)
        self.assertIn("function draw50MACandlestickChart()", self.html_content)
        self.assertIn("function filterScenarioTable(filter)", self.html_content)
        self.assertIn("function render50MAScenarioComparison(scData)", self.html_content)
        self.assertIn("function applyScenarioToLedger(split)", self.html_content)

    def test_07_flask_index_renders_successfully(self):
        """Test 7: Verifies Flask GET / renders status 200 and includes all 50MA UI sections."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('id="tab-ma50-view"', body)
        self.assertIn('id="modal50MAChart"', body)
        self.assertIn('id="canvas50MAChart"', body)

    def test_08_scenario_benchmark_table_and_filters_exist(self):
        """Test 8: Verifies Scenario Comparison Matrix and filter pills are rendered."""
        self.assertIn('id="scenarioTableBody"', self.html_content)
        self.assertIn('id="pillScenarioFilter-ALL"', self.html_content)
        self.assertIn('id="pillScenarioFilter-5Y"', self.html_content)
        self.assertIn('id="pillScenarioFilter-1Y"', self.html_content)
        self.assertIn('id="pillScenarioFilter-MODEL"', self.html_content)
        self.assertIn("Execution Benchmark &amp; Scenario Matrix", self.html_content)
        self.assertIn("All Trades (Last 5 Years)", self.html_content)
        self.assertIn("All Trades (Last 1 Year)", self.html_content)
        self.assertIn("Last 1 Year + AI Model Filter", self.html_content)


if __name__ == "__main__":
    unittest.main()
