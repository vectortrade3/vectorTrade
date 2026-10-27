# 🌐 VectorTrade: Online Execution Micro-Worker Deployment Guide

> 🚀 **Quick Deploy Option (100% Free on Render + UptimeRobot)**:
> If you want to deploy the online worker in 1-click on the cloud for free without managing a Linux server, see the dedicated guide: **[`docs/RENDER_AND_UPTIMEROBOT_DEPLOYMENT.md`](docs/RENDER_AND_UPTIMEROBOT_DEPLOYMENT.md)**.
> A `render.yaml` blueprint is already included in this repository.

This guide details how to host the **VectorTrade Online Micro-Worker** on a low-spec cloud virtual machine (e.g. Google Cloud `e2-micro`, AWS `t4g.nano`, Render, or Oracle Cloud Always Free Ampere) to monitor live price feeds and execute paper trades via MegaBull between 09:17 and 15:32 IST.

The online worker requires **zero heavy dependencies** (no pandas, numpy, scikit-learn, or xgboost) and runs stably with **less than 30 MB of RAM**.

---

## 📦 1. Cloud Host Minimum Requirements

* **RAM:** 256 MB or higher (worker consumes < 30 MB).
* **CPU:** 1 shared vCPU (negligible CPU usage during 2.5s polling loop).
* **OS:** Ubuntu 22.04 LTS or Debian 12.
* **Network:** Stable egress to `api.megabull.in` (port 443) and MongoDB Atlas (port 27017).

---

## ⚙️ 2. Remote Server Setup

### 1. Clone the Repository & Setup Python
```bash
sudo apt-get update && sudo apt-get install -y python3 python3-venv git
git clone <your-repo-url> /opt/vectorTrade
cd /opt/vectorTrade
python3 -m venv venv
source venv/bin/activate
pip install requests pymongo python-dotenv pydantic pydantic-settings pytz
```

### 2. Configure Remote Environment (`.env`)
Create `/opt/vectorTrade/.env` with your production keys:
```ini
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=vectortrade
MEGABULL_API_KEY=your_active_megabull_api_key
MEGABULL_USER_ID=your_megabull_user_id
LOG_LEVEL=INFO
APP_NAME=VectorTrade
TIMEZONE=Asia/Kolkata
```

---

## 🛠 3. Configure systemd Service & Timer

The online worker operates between **09:17 and 15:32 IST (03:47 to 10:02 UTC)**.

### 1. Create `/etc/systemd/system/vectortrade-online.service`
```ini
[Unit]
Description=VectorTrade Online Execution Micro-Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/vectorTrade
EnvironmentFile=/opt/vectorTrade/.env
ExecStart=/opt/vectorTrade/venv/bin/python -m online.execution_daemon
Restart=on-failure
RestartSec=10
MemoryMax=60M
KillMode=process

[Install]
WantedBy=multi-user.target
```

### 2. Create `/etc/systemd/system/vectortrade-online.timer`
```ini
[Unit]
Description=Start VectorTrade Online Worker Daily at 09:17 IST
RefuseManualStart=no
RefuseManualStop=no

[Timer]
# Daily trigger at 09:17 Asia/Kolkata (03:47 UTC)
OnCalendar=*-*-* 03:47:00 UTC
Persistent=true
Unit=vectortrade-online.service

[Install]
WantedBy=timers.target
```

### 3. Enable and Start the Online Schedule
```bash
sudo systemctl daemon-reload
sudo systemctl enable vectortrade-online.timer
sudo systemctl start vectortrade-online.timer
```

---

## 🔍 4. Monitoring & Process Verification

* **Check Service Status:**
  ```bash
  sudo systemctl status vectortrade-online.service
  ```
* **Follow Live Execution Logs:**
  ```bash
  tail -f /opt/vectorTrade/logs/online_execution.log
  ```
* **Audit Active & Realized Positions in MongoDB:**
  ```javascript
  // Open trades
  db.active_trades.find({ status: "OPEN" });

  // Completed trades
  db.trade_history.find().sort({ exit_timestamp: -1 }).limit(10);
  ```
