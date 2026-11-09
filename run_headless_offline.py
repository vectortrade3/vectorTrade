#!/usr/bin/env python3
"""
VectorTrade - Headless Offline Verification Entrypoint
Runs offline diagnostics (data store, models, 50MA strategy, market schedule, MongoDB Atlas sync)
without launching interactive server.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline.app import run_offline_headless

if __name__ == "__main__":
    is_json = "--json" in sys.argv
    result = run_offline_headless(output_format="json" if is_json else "text")
    sys.exit(0 if result.get("success") else 1)
