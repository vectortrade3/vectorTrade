# 🚀 VectorTrade: Offline EOD Daemon Deployment Guide

This guide details how to configure and automate the **VectorTrade Offline End-of-Day (EOD) Daemon** to run automatically post-market (daily strictly at 16:05 IST / 10:35 UTC).

The Offline Daemon is battery-safe and self-terminating: outside its active 2–4 minute run window, it consumes **0% CPU and 0 MB RAM**.

---

## 📋 System Prerequisites

1. **Python 3.10+** environment with required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pydantic pydantic-settings python-dotenv pymongo requests pandas numpy yfinance joblib scikit-learn xgboost
   ```
2. **MongoDB Atlas URI** configured in `.env`:
   ```bash
   MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority
   MEGABULL_API_KEY=your_key_here
   ```

---

## 🐧 Option A: Linux Automated Setup (systemd Timer)

On Linux systems, systemd user timers provide guaranteed execution with persistent catch-up (if the machine was offline at 16:05 IST, it fires immediately on boot).

### 1. Copy Service and Timer Units
```bash
mkdir -p ~/.config/systemd/user
cp systemd/vectortrade-eod.service ~/.config/systemd/user/
cp systemd/vectortrade-eod.timer ~/.config/systemd/user/
```

### 2. Verify Working Directory and Python Path
Ensure `~/.config/systemd/user/vectortrade-eod.service` points to your virtualenv:
```ini
[Service]
Type=oneshot
WorkingDirectory=/path/to/vectorTrade
ExecStart=/path/to/vectorTrade/venv/bin/python -m offline.daily_eod_daemon
```

### 3. Enable and Start the Timer
```bash
systemctl --user daemon-reload
systemctl --user enable vectortrade-eod.timer
systemctl --user start vectortrade-eod.timer
```

### 4. Verify Active Schedule
```bash
systemctl --user list-timers --all
```

---

## 🪟 Option B: Windows Automated Setup (Task Scheduler)

On Windows, use the included PowerShell script to register a scheduled task configured with battery-awareness and wake options.

### 1. Run the Registration Script as Administrator
Open PowerShell as Administrator in the repository folder and execute:
```powershell
powershell -ExecutionPolicy Bypass -File .\systemd\register_windows_task.ps1
```

### 2. Verify Registration
```powershell
Get-ScheduledTask -TaskName "VectorTrade_EOD_Daemon"
```

---

## 🩺 Operational Verification & Logs

* **Local Log Output:** `logs/daemon_runs.log` (automatically rotated when exceeding 10 MB).
* **Database Audit:** Queries `system_audit` collection in MongoDB for execution records:
  ```javascript
  db.system_audit.find({ component: "OFFLINE_EOD_DAEMON" }).sort({ timestamp: -1 }).limit(5)
  ```
* **Manual Dry-Run Check:**
  ```bash
  python -m offline.daily_eod_daemon --dry-run
  ```
