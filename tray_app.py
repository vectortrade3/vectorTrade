"""
VectorTrade - Headless System Tray Desktop Runner
Runs silently in the Windows system tray on ultra-unique port 30307:
1. Headless: No console window shown.
2. Background: Starts the offline Flask application & 4:00 PM auto-scheduler.
3. System Tray: Places a sleek VectorTrade icon in the Windows notification area.
4. Brave Integration: Clicking the tray icon opens http://localhost:30307 directly in Brave Browser (with default browser fallback).
5. Menu Actions: Open Dashboard, Open Orders, Check 4:00 PM Schedule, Exit.
"""

from __future__ import annotations
import os
import sys
import time
import socket
import logging
import threading
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional

# Base directory resolution
if getattr(sys, "frozen", False):
    exe_dir = Path(sys.executable).resolve().parent
    if (exe_dir / "offline").exists():
        PROJECT_ROOT = exe_dir
    elif (exe_dir.parent / "offline").exists():
        PROJECT_ROOT = exe_dir.parent
    elif (exe_dir.parent.parent / "offline").exists():
        PROJECT_ROOT = exe_dir.parent.parent
    else:
        PROJECT_ROOT = Path("D:/Master_Folder/vectorTrade")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Configuration
PORT = 30307
HOST = "127.0.0.1"
CUSTOM_HOST = "vectortrade"
APP_URL = f"http://localhost:{PORT}"
CUSTOM_URL = f"http://{CUSTOM_HOST}:{PORT}"

logger = logging.getLogger("VectorTrade.TrayRunner")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [Tray] %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def find_brave_browser() -> Optional[str]:
    """
    Locates the Brave Browser executable on Windows across standard installation paths.
    """
    candidates = [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        os.path.expandvars(r"%PROGRAMFILES(X86)%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path

    # Check Windows Registry for App Paths
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\brave.exe")
        val, _ = winreg.QueryValueEx(key, "")
        winreg.CloseKey(key)
        if val and os.path.isfile(val):
            return val
    except Exception:
        pass

    return None


def open_in_brave_or_default(url: str = APP_URL) -> bool:
    """
    Opens the target URL in Brave Browser if installed; otherwise falls back to system default.
    """
    brave_path = find_brave_browser()
    if brave_path:
        try:
            logger.info(f"Opening '{url}' in Brave Browser ({brave_path})...")
            # Flags for clean web application window/tab launch
            subprocess.Popen(
                [brave_path, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to launch Brave: {e}. Falling back to default browser.")

    logger.info(f"Opening '{url}' in default browser...")
    return webbrowser.open(url)


def is_port_in_use(port: int = PORT, host: str = HOST) -> bool:
    """Checks whether the server port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def load_tray_icon_image():
    """Loads the icon for the system tray or generates one via Pillow if missing."""
    icon_png = PROJECT_ROOT / "assets" / "vectortrade_icon.png"
    icon_ico = PROJECT_ROOT / "assets" / "vectortrade_icon.ico"

    from PIL import Image, ImageDraw

    if icon_png.exists():
        try:
            return Image.open(str(icon_png))
        except Exception:
            pass

    if icon_ico.exists():
        try:
            return Image.open(str(icon_ico))
        except Exception:
            pass

    # Dynamic fallback icon
    img = Image.new("RGBA", (64, 64), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([2, 2, 62, 62], radius=14, fill=(15, 23, 42, 255), outline=(16, 185, 129, 255), width=2)
    draw.line([(14, 18), (32, 48)], fill=(34, 211, 238, 255), width=6)
    draw.line([(32, 48), (50, 18)], fill=(52, 211, 153, 255), width=6)
    return img


class VectorTradeTrayApp:
    """
    Manages the Windows System Tray lifecycle, background Flask thread, and Brave browser dispatch.
    """

    def __init__(self, port: int = PORT):
        self.port = port
        self._server_thread: Optional[threading.Thread] = None
        self._tray_icon = None
        self._is_running = True

    def start_background_server(self):
        """Launches the VectorTrade Offline Flask App & Scheduler in a background daemon thread."""
        try:
            from offline import json_manager
            json_manager.log_app_event(
                event="APP_STARTED",
                status="Success",
                message=f"VectorTrade application started on port {self.port}."
            )
        except Exception:
            pass

        if is_port_in_use(self.port):
            logger.info(f"Port {self.port} is already active. Assuming existing VectorTrade instance is running.")
            return

        def _run():
            try:
                from offline.app import run_app
                logger.info(f"Starting VectorTrade Offline Engine on port {self.port}...")
                run_app(host="0.0.0.0", port=self.port)
            except Exception as e:
                logger.error(f"Flask server error: {e}", exc_info=True)

        self._server_thread = threading.Thread(
            target=_run,
            daemon=True,
            name="VectorTradeServerThread"
        )
        self._server_thread.start()

        # Brief pause to verify socket readiness
        for _ in range(20):
            if is_port_in_use(self.port):
                logger.info(f"VectorTrade server is successfully listening on port {self.port}.")
                break
            time.sleep(0.1)

    def on_open_dashboard(self, icon=None, item=None):
        """Opens main dashboard in Brave browser."""
        open_in_brave_or_default(f"http://localhost:{self.port}")

    def on_open_orders(self, icon=None, item=None):
        """Opens orders & trade book in Brave browser."""
        open_in_brave_or_default(f"http://localhost:{self.port}/orders")

    def on_open_analysis(self, icon=None, item=None):
        """Opens daily analysis panel in Brave browser."""
        open_in_brave_or_default(f"http://localhost:{self.port}#analysisTab")

    def on_trigger_schedule(self, icon=None, item=None):
        """Triggers manual tick evaluation for the 4:00 PM scheduler."""
        try:
            from offline.scheduler import get_daily_scheduler
            sched = get_daily_scheduler()
            res = sched.run_tick_once(force_execution=False)
            msg = res.get("message", "Schedule evaluated.")
            logger.info(f"Tray scheduler tick: {msg}")
            if self._tray_icon:
                try:
                    self._tray_icon.notify(title="VectorTrade Scheduler", message=msg)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error triggering scheduler: {e}")

    def on_exit(self, icon=None, item=None):
        """Terminates tray application and exits."""
        logger.info("Exiting VectorTrade System Tray Application...")
        try:
            from offline import json_manager
            json_manager.log_app_event(
                event="APP_CLOSED",
                status="Success",
                message="VectorTrade application closed by user from system tray."
            )
        except Exception:
            pass
        self._is_running = False
        if self._tray_icon:
            self._tray_icon.stop()
        os._exit(0)

    def run_tray(self):
        """Initializes and runs the pystray main loop."""
        import pystray

        image = load_tray_icon_image()
        menu = pystray.Menu(
            pystray.MenuItem("Open VectorTrade", self.on_open_dashboard, default=True),
            pystray.MenuItem("Orders & Trade Book", self.on_open_orders),
            pystray.MenuItem("Daily Market Analysis", self.on_open_analysis),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Check 4:00 PM Schedule", self.on_trigger_schedule),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit VectorTrade", self.on_exit)
        )

        self._tray_icon = pystray.Icon(
            name="VectorTrade",
            icon=image,
            title=f"VectorTrade (Port {self.port}) | Click to open in Brave",
            menu=menu
        )

        logger.info(f"VectorTrade System Tray icon ready on port {self.port}. Starting event loop...")
        self._tray_icon.run()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VectorTrade Headless System Tray Desktop Runner")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port number (default: {PORT})")
    parser.add_argument("--open", action="store_true", help="Automatically open browser on launch")
    parser.add_argument("--no-tray", action="store_true", help="Run without system tray icon (headless server only)")
    args = parser.parse_args()

    app = VectorTradeTrayApp(port=args.port)
    app.start_background_server()

    if args.open:
        app.on_open_dashboard()

    if args.no_tray:
        logger.info("Running in headless server-only mode (no tray).")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Terminating...")
            sys.exit(0)
    else:
        app.run_tray()


if __name__ == "__main__":
    main()
