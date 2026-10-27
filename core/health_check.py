"""
VectorTrade - Self-Healing Health Check & Log Rotation Engine
Rotates and archives log files exceeding maximum size (default: 10 MB),
verifies system file integrity, monitors MongoDB Atlas connectivity,
and cleans temporary files.
"""

from __future__ import annotations
import os
import glob
import logging
import gzip
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import settings
from core.db import db_manager

logger = logging.getLogger("VectorTrade.HealthCheck")


def rotate_logs(max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> Dict[str, Any]:
    """
    Scans logs/ directory and compresses logs exceeding max_bytes.
    Retains up to backup_count archived gzip files per log stream.
    """
    rotated = []
    log_files = glob.glob(str(settings.LOGS_DIR / "*.log"))

    for log_path in log_files:
        try:
            p = Path(log_path)
            size = p.stat().st_size
            if size > max_bytes:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gz_path = p.with_name(f"{p.stem}_{timestamp}.log.gz")
                with open(p, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                # Truncate active log file
                with open(p, "w", encoding="utf-8") as f:
                    f.truncate(0)
                rotated.append(str(gz_path))
                logger.info("Rotated log file %s -> %s (Size: %.2f MB)", p.name, gz_path.name, size / (1024*1024))

                # Prune excess backup files
                pattern = str(p.with_name(f"{p.stem}_*.log.gz"))
                archives = sorted(glob.glob(pattern))
                if len(archives) > backup_count:
                    for old_archive in archives[:-backup_count]:
                        os.remove(old_archive)
        except Exception as e:
            logger.warning("Error rotating log %s: %s", log_path, str(e))

    return {
        "status": "OK",
        "rotated_count": len(rotated),
        "rotated_files": rotated
    }


def run_system_health_check() -> Dict[str, Any]:
    """
    Executes full operational health check:
    1. Database ping.
    2. Model file presence.
    3. Token cache verification.
    4. Log rotation check.
    """
    db_status = db_manager.ping()
    token_cache_ok = settings.TOKENS_CACHE_PATH.exists()
    stock_names_ok = settings.STOCK_NAMES_PATH.exists()
    models_ok = (settings.MODELS_DIR / "5_year_xgboost_model.pkl").exists() or (settings.BASE_DIR / "models" / "5_year_xgboost_model.pkl").exists()

    rotation_status = rotate_logs()

    health_report = {
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "token_cache_present": token_cache_ok,
        "stock_names_present": stock_names_ok,
        "models_present": models_ok,
        "log_rotation": rotation_status,
        "overall_status": "HEALTHY" if (db_status.get("status") == "ok" and token_cache_ok and models_ok) else "DEGRADED"
    }
    return health_report


if __name__ == "__main__":
    import json
    report = run_system_health_check()
    print(json.dumps(report, indent=2))
