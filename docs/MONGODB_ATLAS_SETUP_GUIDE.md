# 🍃 MongoDB Atlas Setup Guide for VectorTrade

MongoDB Atlas acts as the secure, high-speed asynchronous bridge between:
1. **Your Offline Desktop Machine** (Generates indicators at 16:00 IST and pushes trade candidates).
2. **Your Online Cloud Engine** (Polls tick data from 9:17 AM – 3:32 PM IST, executes paper orders, and guards open trades).

Atlas offers a **100% Free Forever Tier (M0 Cluster, 512 MB storage, no credit card required)**, which is more than enough to store years of trade setups, historical audit logs, and performance metrics.

---

## 📋 Step-by-Step Instructions to Get Credentials

### Step 1: Create Free Atlas Account
1. Visit: **[https://www.mongodb.com/cloud/atlas/register](https://www.mongodb.com/cloud/atlas/register)**
2. Sign up with your Google account or email address (free, no credit card required).

---

### Step 2: Create Free Shared Cluster (M0)
1. In the MongoDB Atlas console, click **Create Database** / **Build a Database**.
2. Select the **M0 Shared** (Free Forever) option:
   * **Cloud Provider**: AWS or Google Cloud.
   * **Region**: Choose a region near you (e.g., **Mumbai `ap-south-1`** for the lowest latency in India).
   * **Cluster Name**: `Cluster0` (or leave default).
3. Click **Create Cluster**.

---

### Step 3: Create Database User Credentials
During the quickstart setup (or under **Security > Database Access** in the left menu):
1. Click **Add New Database User**.
2. **Authentication Method**: Password.
3. **Username**: Enter a username (e.g. `vectortrade_admin`).
4. **Password**: Click **Autogenerate Secure Password** or enter your own (e.g. `VectorTradePass2026`).
   > ⚠️ **Important**: Copy this password to a notepad! You will need it in Step 6.
5. **Database User Privileges**: Select **Read and write to any database** (or Atlas admin).
6. Click **Add User**.

---

### Step 4: Configure Network IP Whitelist
*(Under **Security > Network Access** in the left menu)*:
1. Click **Add IP Address**.
2. Select **Allow Access From Anywhere** (this enters `0.0.0.0/0`).
3. Add a description: `VectorTrade Home and Cloud Access`.
4. Click **Confirm**.
   > 💡 **Why `0.0.0.0/0`?** Because your home ISP IP address changes frequently (dynamic IP) and your cloud VM running the online worker has a different IP. This allows both machines to communicate seamlessly.

---

### Step 5: Get Python Connection String
1. Go to **Deployment > Database** in the left sidebar.
2. Click the **Connect** button next to your cluster.
3. Choose **Drivers** under "Connect to your application".
4. Set:
   * **Driver**: `Python`
   * **Version**: `3.11 or later`
5. You will see a connection string like this:
   ```text
   mongodb+srv://vectortrade_admin:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
6. Click the **Copy** icon.

---

### Step 6: Paste into `.env`
Open your `.env` file in `D:\Master_Folder\vectorTrade\.env` and configure:

```env
# MongoDB Atlas Database URI
# Replace <password> with the password you saved in Step 3
MONGODB_URI=mongodb+srv://vectortrade_admin:VectorTradePass2026@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=vectortrade
```

> ⚠️ Note: If your password contains special characters (like `@`, `:`, `/`, `%`), URL-encode them or use an alphanumeric password (e.g., letters and numbers only) to avoid parsing issues.

---

### Step 7: Verify Connection
Run this simple command in your terminal to verify that your cluster is connected:

```powershell
python -c "from core.db import db_manager; print(db_manager.ping())"
```

If successful, you will see:
```text
{'status': 'ok', 'mode': 'atlas_cluster', 'database': 'vectortrade'}
```

---

## 🛡️ Fallback Guarantee
If `MONGODB_URI` is left as `mock` or empty in `.env`:
* VectorTrade automatically switches to **in-memory mock mode** (`mongomock`).
* Both Online and Offline systems continue to run locally for development and testing without needing an internet connection.
