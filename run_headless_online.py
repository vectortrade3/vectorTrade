#!/usr/bin/env python3
"""
VectorTrade - Headless Online Verification Entrypoint
Runs online application diagnostics (MongoDB Atlas, Orders service, MegaBull REST broker telemetry,
and full HTTP route suite) without launching interactive server.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from online.app import run_online_headless

if __name__ == "__main__":
    is_json = "--json" in sys.argv
    result = run_online_headless(output_format="json" if is_json else "text")
    sys.exit(0 if result.get("success") else 1)
