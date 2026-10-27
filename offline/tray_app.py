"""
VectorTrade - Production Windows System Tray Application
Runs silently in the Windows notification area with 0 console distractions.
Hosts the background scheduler and local Flask control dashboard on http://127.0.0.1:5000.
"""

from __future__ import annotations
import sys
import os
import threading
import webbrowser
import logging
from pathlib import Path
from PIL import Image, ImageDraw

# Root directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent
OFFLINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(OFFLINE_DIR))

import pystray
from pystray import MenuItem as item, Menu

from core.config import settings
from offline.scheduler import offline_scheduler
from offline.server import run_server

logger = logging.getLogger("VectorTrade.TrayApp")


def create_tray_icon_image() -> Image.Image:
    """
    Creates a slick 64x64 RGBA vector icon with 'V' branding programmatically
    so the app never crashes due to missing external image files.
    """
    width, height = 64, 64
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dc = ImageDraw.Draw(image)

    # Rounded dark slate background
    dc.rounded_rectangle([2, 2, width - 2, height - 2], radius=14, fill=(15, 23, 42, 255), outline=(245, 158, 11, 255), width=2)

    # Draw vibrant amber 'V' chevron
    points = [
        (16, 18),
        (24, 18),
        (32, 42),
        (40, 18),
        (48, 18),
        (36, 48),
        (28, 48)
    ]
    dc.polygon(points, fill=(245, 158, 11, 255))

    return image


class VectorTradeTrayApp:
    """
    Windows System Tray Controller managing background daemons and Flask web server.
    """

    def __init__(self, port: int = 5000) -> None:
        self.port = port
        self.icon: pystray.Icon | None = None
        self.server_thread: threading.Thread | None = None
        self.scheduler = offline_scheduler

    def open_dashboard(self, icon=None, item=None) -> None:
        """Launches the default web browser to the offline control center."""
        webbrowser.open(f"http://127.0.0.1:{self.port}")

    def open_online_hub(self, icon=None, item=None) -> None:
        """Launches the default web browser to the online execution hub."""
        webbrowser.open(settings.ONLINE_BASE_URL)

    def trigger_eod_manual(self, icon=None, item=None) -> None:
        """Runs EOD feature generation and strategy inference in a worker thread."""
        threading.Thread(target=self.scheduler.run_eod_pipeline, kwargs={"force": True}, daemon=True).start()

    def trigger_sync_manual(self, icon=None, item=None) -> None:
        """Flushes queued local dispatches to MongoDB Atlas."""
        threading.Thread(target=self.scheduler.sync_dispatches, daemon=True).start()

    def get_status_text(self, item=None) -> str:
        """Dynamic text displayed on the tray menu."""
        state = self.scheduler.get_system_status()
        status_name = "Busy (Running Analysis...)" if state["is_busy"] else f"Status: Idle (Next: {state['next_scheduled_run'].split(' ')[1]})"
        return status_name

    def quit_app(self, icon, item=None) -> None:
        """Gracefully shuts down scheduler and removes tray icon."""
        logger.info("Stopping VectorTrade Tray Application...")
        self.scheduler.stop()
        icon.stop()
        # Force terminate process cleanly
        os._exit(0)

    def build_menu(self) -> Menu:
        return Menu(
            item("🚀 Open VectorTrade Dashboard", self.open_dashboard, default=True),
            item("⚡ Run EOD Analysis Now", self.trigger_eod_manual),
            item("🔄 Sync Pending Dispatches", self.trigger_sync_manual),
            item("🌐 Open Online Hub (Port 8000)", self.open_online_hub),
            Menu.SEPARATOR,
            item(self.get_status_text, lambda icon, item: None, enabled=False),
            Menu.SEPARATOR,
            item("❌ Exit VectorTrade", self.quit_app)
        )

    def run(self) -> None:
        """Starts background servers and enters pystray event loop."""
        logger.info("Initializing VectorTrade Offline Tray App...")

        # 1. Start background automation scheduler
        self.scheduler.start()

        # 2. Start Flask Web Server in daemon thread
        self.server_thread = threading.Thread(
            target=run_server,
            kwargs={"host": "127.0.0.1", "port": self.port},
            daemon=True,
            name="VectorTrade-FlaskServer"
        )
        self.server_thread.start()

        # 3. Create Tray Icon
        icon_img = create_tray_icon_image()
        self.icon = pystray.Icon(
            name="VectorTrade",
            icon=icon_img,
            title="VectorTrade — Offline Engine Active",
            menu=self.build_menu()
        )

        logger.info("VectorTrade resident in System Tray. Dashboard available at http://127.0.0.1:%d", self.port)
        self.icon.run()


def main():
    app = VectorTradeTrayApp(port=5000)
    app.run()


if __name__ == "__main__":
    main()
