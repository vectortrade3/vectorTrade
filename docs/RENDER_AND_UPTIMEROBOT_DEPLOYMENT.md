# 🚀 VectorTrade: Render Cloud + UptimeRobot Deployment Guide

Deploying the **VectorTrade Online Engine & Live Execution Hub** on **Render** paired with **UptimeRobot** is an **ideal, production-ready, and 100% FREE architecture**.

---

## 💡 Is Render + UptimeRobot a Good Strategy?

### **YES, absolutely! Here is why:**

1. **Zero Cloud Hosting Costs**:
   * Render provides **750 free instance hours per month**.
   * One full month has 744 hours (31 days × 24 hours).
   * **744 hours < 750 free hours** → A single service can run **24/7 completely free** every single month!
2. **Solving Render's 15-Minute Sleep Mode**:
   * On Render's Free Web Service, containers spin down (sleep) after 15 minutes of inactivity.
   * If Render sleeps, the background trading engine stops polling market prices.
   * **UptimeRobot completely solves this**: It pings your Render URL every **5 minutes**, which ensures Render **NEVER goes to sleep**!
3. **Low-Resource Footprint**:
   * Our online worker does not load pandas, numpy, scikit-learn, or xgboost.
   * Total RAM usage is **< 30 MB** (Render gives you 512 MB).
   * The build completes in just **10–15 seconds**.
4. **Crash & Restart Resilience**:
   * Because VectorTrade's online engine is stateless locally and stores all active positions in MongoDB Atlas, if Render ever restarts for container maintenance, it boots back up in **< 2 seconds**, reloads all open trades from Atlas, and resumes monitoring with **zero data loss**.

> ⚠️ **One Rule to Keep It Free**: Do **not** run multiple web services on the same free Render account, as they will split the 750 free hours. Only keep this 1 service active on your free Render account.

---

## 🛠️ Part 1: Deploying Only the Online App on Render

You do **not** need to split your repository into multiple repos. Render allows you to build and run only the `online/` package from your single GitHub repository.

### Step 1: Upload Your Code to GitHub
Make sure your `.gitignore` is up to date (we have already updated it to exclude `.env`, `.exe`, logs, datasets, and pickle models).
Push your repository to GitHub:
```bash
git add .
git commit -m "feat: VectorTrade full production release"
git push origin main
```

---

### Step 2: Create a Web Service on Render
1. Go to **[https://render.com](https://render.com)** and sign in (using GitHub).
2. Click **New +** in the top right and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your `VectorTrade` repository.

---

### Step 3: Configure Service Settings
Fill in the following exact settings:

| Setting | Value |
|---|---|
| **Name** | `vectortrade-online` |
| **Region** | `Singapore` *(nearest to NSE India for lowest latency)* |
| **Branch** | `main` |
| **Root Directory** | *(Leave blank / empty)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r online/requirements_online.txt` |
| **Start Command** | `uvicorn online.app:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` ($0/month) |

---

### Step 4: Add Environment Variables in Render
Scroll down to the **Environment Variables** section and add the following keys:

| Key | Value | Description |
|---|---|---|
| `PYTHONPATH` | `.` | Ensures Python finds `core/` and `online/` modules |
| `MONGODB_URI` | `mongodb+srv://<user>:<pass>@cluster0.xxx.mongodb.net/?retryWrites=true&w=majority` | Your MongoDB Atlas connection string |
| `MONGODB_DB_NAME` | `vectortrade` | MongoDB Database Name |
| `MEGABULL_API_KEY` | `your_megabull_api_key` | MegaBull API key from your profile |
| `MEGABULL_USER_ID` | `your_user_id` | Optional user ID |
| `INTERNAL_AUTH_TOKEN` | `vt_sec_789f2a4bc81d45e69b03e2c4d8e1a7` | Secret key for secure offline-online comms |
| `TIMEZONE` | `Asia/Kolkata` | Guarantees IST market hours |
| `APP_NAME` | `VectorTrade` | App Name |
| `LOG_LEVEL` | `INFO` | Logging level |

---

### Step 5: Click "Create Web Service"
1. Click **Create Web Service** at the bottom.
2. Render will clone the repo, install the lightweight online dependencies (~15 seconds), and start the server.
3. Once deployed, Render will give you a public URL like:
   `https://vectortrade-online.onrender.com`
4. Click the link to open your live quantitative dashboard!

---

## ⏱️ Part 2: Setting up UptimeRobot (Keep Awake 24/7)

### Step 1: Create Free UptimeRobot Account
1. Go to **[https://uptimerobot.com](https://uptimerobot.com)**.
2. Sign up for a free account (allows 50 monitors with 5-minute intervals for free).

---

### Step 2: Create a Keep-Alive Monitor
1. On the UptimeRobot dashboard, click **+ Add New Monitor**.
2. Configure the monitor:
   * **Monitor Type**: Select `HTTP(s)`
   * **Friendly Name**: `VectorTrade Online Keeper`
   * **URL (or IP)**: Paste your Render URL:
     ```text
     https://vectortrade-online.onrender.com/
     ```
   * **Monitoring Interval**: Select `Every 5 minutes`
3. Click **Create Monitor**.

---

### 🎉 Done! What Happens Now?
* UptimeRobot will ping your Render online server every 5 minutes.
* Render detects active HTTP traffic and **never spins down**.
* Between **9:17 AM and 3:32 PM IST**, your online execution worker will continuously query MegaBull LTPs, enforce Day-1 invalidations, and protect your open positions with automated SL and Target exits.
* You can open `https://vectortrade-online.onrender.com/` on your phone, tablet, or work laptop anytime to view live trades in real time.
