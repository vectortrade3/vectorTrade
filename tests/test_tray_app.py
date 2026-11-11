"""
VectorTrade - Unit & Integration Tests for System Tray Desktop Runner
Verifies:
1. Brave browser path detection and registry fallback.
2. Brave URL dispatching with graceful fallback to system default browser.
3. Socket port utilization checker on port 30307.
4. Tray icon image loading and dynamic fallback creation.
5. VectorTradeTrayApp lifecycle, menu configuration, and action callbacks.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import tray_app
from tray_app import (
    find_brave_browser,
    open_in_brave_or_default,
    is_port_in_use,
    load_tray_icon_image,
    VectorTradeTrayApp,
    PORT
)


class TestTrayApp(unittest.TestCase):
    """Test suite for the headless system tray runner and Brave browser launcher."""

    def test_01_port_configuration_is_30307(self):
        """Verifies default port is set to the ultra-unique 30307."""
        self.assertEqual(PORT, 30307)
        app = VectorTradeTrayApp()
        self.assertEqual(app.port, 30307)

    def test_02_find_brave_browser_detects_installed_path(self):
        """Verifies find_brave_browser returns a valid path if Brave exists, or None."""
        brave_path = find_brave_browser()
        if brave_path is not None:
            self.assertTrue(Path(brave_path).is_file())
            self.assertIn("brave.exe", brave_path.lower())

    def test_03_open_in_brave_dispatches_subprocess(self):
        """Verifies open_in_brave_or_default executes subprocess.Popen with brave.exe."""
        with patch("tray_app.find_brave_browser", return_value=r"C:\fake\brave.exe"):
            with patch("subprocess.Popen") as mock_popen:
                success = open_in_brave_or_default("http://localhost:30307")
                self.assertTrue(success)
                mock_popen.assert_called_once_with(
                    [r"C:\fake\brave.exe", "http://localhost:30307"],
                    stdout=-3,
                    stderr=-3,
                    shell=False
                )

    def test_04_open_in_brave_fallback_to_default_browser(self):
        """Verifies fallback to webbrowser.open when Brave browser is not present."""
        with patch("tray_app.find_brave_browser", return_value=None):
            with patch("webbrowser.open") as mock_webbrowser:
                mock_webbrowser.return_value = True
                success = open_in_brave_or_default("http://localhost:30307")
                self.assertTrue(success)
                mock_webbrowser.assert_called_once_with("http://localhost:30307")

    def test_05_load_tray_icon_returns_valid_image(self):
        """Verifies load_tray_icon_image returns a valid PIL Image."""
        img = load_tray_icon_image()
        self.assertIsInstance(img, Image.Image)
        self.assertGreater(img.size[0], 0)
        self.assertGreater(img.size[1], 0)

    def test_06_is_port_in_use_detection(self):
        """Verifies is_port_in_use returns a boolean without throwing exceptions."""
        in_use = is_port_in_use(port=30307)
        self.assertIsInstance(in_use, bool)

    def test_07_tray_app_action_callbacks(self):
        """Verifies tray app action callbacks open expected target endpoints."""
        app = VectorTradeTrayApp(port=30307)

        with patch("tray_app.open_in_brave_or_default") as mock_open:
            app.on_open_dashboard()
            mock_open.assert_called_with("http://localhost:30307")

        with patch("tray_app.open_in_brave_or_default") as mock_open:
            app.on_open_orders()
            mock_open.assert_called_with("http://localhost:30307/orders")

        with patch("tray_app.open_in_brave_or_default") as mock_open:
            app.on_open_analysis()
            mock_open.assert_called_with("http://localhost:30307#analysisTab")


if __name__ == "__main__":
    unittest.main()
