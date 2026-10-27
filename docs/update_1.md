# 🚀 NIFTY 200 Hybrid Algorithmic Trading Architecture (Update 1)

This document specifies the end-to-end production architecture, deployment model, and implementation phases for evolving the NIFTY 200 ML Trading Strategy Pipeline from research notebooks into a resilient, decoupled paper-trading factory.

---

## 🏛 1. High-Level Architecture Overview

```
                      +---------------------------------------+
                      |         LOCAL SYSTEM (Offline)        |
                      |  - Event-driven run (systemd / task)  |
                      |  - Runs strictly post 16:00 IST       |
                      |  - 0% RAM / 0% CPU outside run window |
                      |  - yfinance EOD ingestion             |
                      |  - 130+ feature calculations          |
                      |  - Pluggable Strategy Engines         |
                      |  - ML Inference (.pkl Brains)         |
                      |  - Monthly retrainer (1st of month)   |
                      +-------------------+-------------------+
                                          |
                      Writes Signals &    | Reads trade outcomes
                      Monthly Buffer Bars | for model retraining
                                          v
                      +---------------------------------------+
                      |       DATABASE BUS (MongoDB Atlas)     |
                      |  - signals_watchlist (Armed triggers) |
                      |  - active_trades (Live monitoring)    |
                      |  - trade_history (Realized PnL / MAE) |
                      |  - buffer_daily_bars (Monthly cache)  |
                      |  - system_audit_logs (Run verification)|
                      +-------------------+-------------------+
                                          ^
                      Reads Armed Setups  | Writes order fills &
                      during market hours | PnL real-time exits
                                          |
                      +-------------------+-------------------+
                      |      CLOUD / MICRO-RUNNER (Online)    |
                      |  - Runs 09:17 to 15:32 IST            |
                      |  - Zero ML / No pandas / RAM < 40 MB  |
                      |  - MegaBull REST Polling (2-3s)       |
                      |  - Day-1 Pre-entry Invalidation check |
                      |  - Bracket Target/SL execution        |
                      +-------------------+-------------------+
                                          |
                                          | Live REST API Calls
                                          v
                      +---------------------------------------+
                      |       MEGABULL BROKER ENGINE          |
                      |  - POST /api/order/buysell            |
                      |  - GET  /api/marketwatch/ltp/{token}  |
                      |  - GET  /api/marketwatch/instruments  |
                      |  - GET  /api/position/my              |
                      +---------------------------------------+
```

---

## 📂 2. Target Project Directory Structure

```text
nifty200-live-engine/
├── .env                              # Sensitive API keys, Mongo URI (untracked)
├── .env.example                      # Dummy configuration template
├── .gitignore                        # Standard Python, models, logs, and env exclusions
├── README.md                         # Operational manual and runbook
├── config/
│   ├── __init__.py
│   ├── settings.py                   # Environment variable loader and path constants
│   └── mongo_client.py               # Centralized PyMongo connection pool
├── core/
│   ├── __init__.py
│   ├── indicators.py                 # Pure calculation of 130+ technical indicators
│   ├── token_manager.py              # MegaBull instruments downloader and symbol mapper
│   └── strategies/
│       ├── __init__.py
│       ├── base_strategy.py          # Abstract strategy interface (Scalable base)
│       ├── ma50_pullback.py          # Strategy 1: 50MA + BB Sniper (Setups 1, 2, 3)
│       └── global_forecast.py        # Strategy 2: 7-Day Max/Min Forecaster
├── offline/
│   ├── __init__.py
│   ├── runner.py                     # Entry point for the EOD local engine
│   ├── data_ingestion.py             # Incremental yfinance daily bar fetcher
│   ├── inference_engine.py           # Loads .pkl brains and calculates probabilities
│   ├── monthly_retrainer.py          # Retrains models on 1st of month with closed trade outcomes
│   ├── logger.py                     # Dual-stream logging (local file + MongoDB audit)
│   └── models/                       # Versioned model weights (.pkl)
├── online/
│   ├── __init__.py
│   ├── runner.py                     # Entry point for the lightweight cloud executor
│   ├── megabull_api.py               # Minimalist REST client for MegaBull
│   ├── state_machine.py              # Order state transitions (ARMED -> TRIGGERED / INVALIDATED)
│   └── position_tracker.py           # Real-time SL / Target exit monitor
└── systemd/                          # Production Linux background service files
    ├── nifty-offline.service
    ├── nifty-offline.timer
    ├── nifty-online.service
    └── nifty-online.timer
```

---

## 📊 3. Scalable Strategy Rules & Math Formulations

### Strategy 1: 50MA Pullback ("Sniper")
* **Base Filter Conditions:**
  * 50MA trend is strictly positive: $\text{slope}(\text{MA50}) > 0$ for at least 7 consecutive days.
  * Price proximity to Lower Bollinger Band: $-10 \le (\text{Low} - \text{BB}_{\text{lower}}) \le 20$.
  * Candle structure is bullish: $\text{Close} > \text{Open}$.
* **Classification into 3 Sub-Setups:**
  1. **Setup 1 (Confluence Zone):** $-10 \le (\text{Low} - \text{MA50}) \le 30$.
  2. **Setup 2 (Extended Trend):** $(\text{Low} - \text{MA50}) > 30$.
  3. **Setup 3 (Mean-Reverting Under 50MA):** $(\text{Low} - \text{MA50}) < -10$.
* **Execution Bracket Formulas:**
  $$\text{Buy Trigger} = \text{High}_{\text{signal}} + 1.0$$
  $$\text{Stop Loss (SL)} = \text{Low}_{\text{signal}} - 1.0$$
  $$\text{Risk} = \text{Buy Trigger} - \text{SL}$$
  $$\text{Target} = \text{Buy Trigger} + (2 \times \text{Risk})$$
* **Day-1 Execution Priority Rule:**
  * If the market opens/moves such that $\text{LTP} \le \text{SL}$ *before* $\text{LTP} \ge \text{Buy Trigger}$, the signal is declared **INVALIDATED** and discarded immediately.
  * If $\text{LTP} \ge \text{Buy Trigger}$ triggers first, execute Buy and monitor Target / SL exits for up to 7 trading days.

### Strategy 2: Global Forecaster
* Evaluates all 200 constituent stocks daily.
* Two regression models predict:
  1. $G_{\max}$: Maximum percentage gain over the next 7 trading days.
  2. $D_{\max}$: Maximum percentage drop over the next 7 trading days.
* Generates an actionable signal if $\frac{G_{\max}}{|D_{\max}|} \ge 2.0$ with a minimum threshold on $G_{\max} \ge 3.5\%$.

---

## 🗄 4. MongoDB Schema Specifications

### Collection: `signals_watchlist`
Stores prospective trade setups generated by the offline pipeline for the upcoming market session.
```json
{
  "_id": "TATAMOTORS_20261026_50MA",
  "symbol": "TATAMOTORS",
  "instrument_token": "3456",
  "strategy_id": "50MA_PULLBACK",
  "setup_type": 1,
  "ml_confidence": 0.824,
  "action": "BUY",
  "buy_trigger": 952.00,
  "sl_price": 939.00,
  "target_price": 978.00,
  "quantity": 50,
  "max_hold_days": 7,
  "status": "ARMED",
  "valid_for_date": "2026-10-26",
  "created_at": "2026-10-25T16:15:00Z"
}
```
*Statuses:* `ARMED`, `TRIGGERED`, `INVALIDATED`, `EXPIRED`.

### Collection: `active_trades`
Stores live positions actively being monitored by the online worker.
```json
{
  "_id": "TRADE_MB_883912",
  "symbol": "TATAMOTORS",
  "instrument_token": "3456",
  "strategy_id": "50MA_PULLBACK",
  "megabull_order_id": "MB-ORD-20261026-0012",
  "fill_price": 952.25,
  "sl_price": 939.00,
  "target_price": 978.00,
  "quantity": 50,
  "days_active": 1,
  "max_hold_days": 7,
  "entry_timestamp": "2026-10-26T09:48:12Z",
  "status": "OPEN"
}
```

### Collection: `trade_history`
Maintains permanent audit records for completed trades.
```json
{
  "symbol": "TATAMOTORS",
  "strategy_id": "50MA_PULLBACK",
  "entry_price": 952.25,
  "exit_price": 978.00,
  "pnl": 1287.50,
  "roi_pct": 2.70,
  "exit_reason": "TARGET_HIT",
  "entry_date": "2026-10-26",
  "exit_date": "2026-10-28",
  "days_held": 3,
  "mfe": 28.50,
  "mae": -3.20,
  "ml_outcome_label": 1
}
```

### Collection: `buffer_daily_bars`
Accumulates full feature rows daily. Flushed on the 1st of every month to retrain the ML brains.
```json
{
  "date": "2026-10-26",
  "symbol": "TATAMOTORS",
  "features": {
    "close": 950.10,
    "ma50": 924.30,
    "rsi14": 46.2,
    "bb_percent_b": 0.12
  },
  "consumed_for_training": false
}
```

### Collection: `system_audit_logs`
Records the dual-check execution log.
```json
{
  "component": "OFFLINE_DAEMON",
  "execution_date": "2026-10-26",
  "timestamp": "2026-10-26T16:04:12Z",
  "status": "SUCCESS",
  "symbols_ingested": 200,
  "signals_detected": 4,
  "db_push_acknowledged": true,
  "duration_seconds": 18.4
}
```

---

## 🛠 5. Implementation Phases & Step-by-Step Milestones

---

### Phase 1: Environment Isolation & Unified Infrastructure Setup
*Objective:* Establish the independent project structure, separate configuration from code, and initialize the MongoDB Atlas bus.

* **Step 1.1:** Initialize the new dedicated directory and setup Python virtual environment (`python -m venv venv`).
* **Step 1.2:** Implement `config/settings.py` utilizing `pydantic-settings` or `python-dotenv` to validate environment keys (`MONGODB_URI`, `MEGABULL_API_KEY`, `MEGABULL_USER_ID`).
* **Step 1.3:** Build `config/mongo_client.py` with thread-safe singleton connection pooling and automatic index creation (`valid_for_date`, `status`, `instrument_token`).
* **Step 1.4:** Create `core/token_manager.py` to pull instrument definitions from `GET https://api.megabull.in/api/marketwatch/instruments`, parse the CSV download URL, and generate a local cache `tokens_cache.json` matching NSE symbols to MegaBull tokens.

> **Verification: <PHASE_1_COMPLETED>**
> 1. Run test script `python -m core.token_manager` confirming `tokens_cache.json` is generated with 200 valid NIFTY tokens.
> 2. Execute a database ping via `mongo_client.py` confirming successful write and read of a test document into MongoDB Atlas.
> 3. Zero hardcoded secrets verified by running a git check.

---

### Phase 2: Offline Analytics & Strategy Dispatcher
*Objective:* Package the notebook feature engineering into production modules, implement pluggable strategy classes, and construct the EOD runner.

* **Step 2.1:** Port feature calculations from `5_years_2_feature_generation.ipynb` into a vectorized module `core/indicators.py`.
* **Step 2.2:** Build the abstract base class `core/strategies/base_strategy.py` with mandatory methods `generate_signals(df) -> list[dict]`.
* **Step 2.3:** Implement `core/strategies/ma50_pullback.py` with strict Setups 1, 2, and 3 categorization and exact bracket pricing formulas.
* **Step 2.4:** Implement `core/strategies/global_forecast.py` to evaluate 7-day high/low forecasts.
* **Step 2.5:** Build `offline/inference_engine.py` to load `.pkl` models from `offline/models/` and filter raw signals.
* **Step 2.6:** Implement `offline/runner.py` with execution guards:
  * Check current time (must be $\ge 16:00$ IST).
  * Check `system_audit_logs` in MongoDB: If an entry exists for `execution_date == TODAY` and `status == "SUCCESS"`, terminate immediately (`sys.exit(0)`).
  * Fetch EOD daily bar for 200 symbols via `yfinance`.
  * Calculate features, run strategy filters, run ML predictions.
  * Push qualified setups to `signals_watchlist` with status `ARMED`.
  * Append feature data to `buffer_daily_bars`.
  * Log execution summary to `system_audit_logs` and local file `logs/offline_run.log`.

> **Verification: <PHASE_2_COMPLETED>**
> 1. Execute `offline/runner.py` manually during simulation mode. Verify `system_audit_logs` receives a `SUCCESS` record.
> 2. Verify that running `offline/runner.py` a second time immediately exits in $< 0.5$ seconds with log message `SKIPPED_ALREADY_RAN_TODAY`.
> 3. Inspect MongoDB `signals_watchlist` and verify candidate documents contain exact `buy_trigger`, `sl_price`, `target_price`, and `instrument_token`.

---

### Phase 3: Zero-Idle Battery-Safe Daemon Integration
*Objective:* Configure local operating system scheduling so no continuous Python loop occupies memory or battery outside market hours.

* **Step 3.1:** Write the systemd user service `nifty-offline.service` referencing the virtual environment Python and `offline/runner.py`.
* **Step 3.2:** Write the systemd user timer `nifty-offline.timer` set to fire daily at `16:05 IST` with `Persistent=true` (guaranteeing execution on startup if the machine was off at 16:05).
* **Step 3.3:** Create an auxiliary Windows Task Scheduler XML profile for cross-platform portability.

> **Verification: <PHASE_3_COMPLETED>**
> 1. Run `systemctl --user status nifty-offline.timer` confirming active next trigger timestamp.
> 2. Verify via `ps aux | grep python` that no persistent Python daemon runs while idle (RAM consumption is 0 MB).

---

### Phase 4: Online Execution Micro-Worker (MegaBull REST)
*Objective:* Build the lightweight, stateful execution engine that interacts with MegaBull from 09:17 to 15:32 IST.

* **Step 4.1:** Build `online/megabull_api.py` handling REST calls:
  * Headers: `x-api-key: {MEGABULL_API_KEY}` and `user-id: {MEGABULL_USER_ID}`.
  * `get_ltp(token)` -> queries `GET /api/marketwatch/ltp/{token}`.
  * `place_order(symbol, token, action, quantity, price, order_type)` -> `POST /api/order/buysell`.
  * `get_positions()` -> `GET /api/position/my`.
* **Step 4.2:** Build `online/state_machine.py` managing the trade lifecycle:
  * **Invalidation Gate:** If $\text{LTP} \le \text{SL}$ before $\text{LTP} \ge \text{Buy Trigger}$, update status in `signals_watchlist` to `INVALIDATED` and drop from monitoring.
  * **Entry Trigger:** If $\text{LTP} \ge \text{Buy Trigger}$, send Buy order to MegaBull, retrieve order ID, create document in `active_trades`, set status in `signals_watchlist` to `TRIGGERED`.
* **Step 4.3:** Build `online/position_tracker.py`:
  * Monitors open documents in `active_trades`.
  * If $\text{LTP} \ge \text{Target}$, trigger Sell order, record `TARGET_HIT` in `trade_history`, remove from `active_trades`.
  * If $\text{LTP} \le \text{SL}$, trigger Sell order, record `SL_HIT` in `trade_history`, remove from `active_trades`.
* **Step 4.4:** Build `online/runner.py`:
  * Starts at 09:17 IST.
  * Pulls `ARMED` signals from MongoDB.
  * Runs a 2.5-second polling loop over required tokens.
  * Reconciles open positions at 15:30 IST.
  * At 15:32 IST, marks remaining untriggered `ARMED` signals as `EXPIRED` and exits.

> **Verification: <PHASE_4_COMPLETED>**
> 1. Execute `online/runner.py` with mock armed signals. Test the Day-1 invalidation path (simulate LTP touching SL first) and confirm MongoDB updates to `INVALIDATED` without placing an order.
> 2. Test entry trigger path on a test symbol; verify successful paper order creation on MegaBull via `POST /api/order/buysell`.
> 3. Verify target and stop loss triggers correctly send an exit order and persist realized PnL to `trade_history`.
> 4. Verify process memory footprint remains below 40 MB under full polling load.

---

### Phase 5: Feedback Loop & Automated Monthly Retraining
*Objective:* Connect closed trade performance directly into the monthly model retraining pipeline.

* **Step 5.1:** Implement `offline/monthly_retrainer.py`.
* **Step 5.2:** Add date evaluation: If today's date is the 1st of the month:
  * Pull all daily bar records from `buffer_daily_bars` accumulated over the past month.
  * Pull completed trades from `trade_history` to label real-world outcomes.
  * Append verified samples to master training dataset (`final_trade_data.csv`).
  * Trigger retraining for Strategy 1 (Sniper classifier) and Strategy 2 (Global Forecaster).
  * Validate that new model AUC/Accuracy meets baseline standards before replacing existing `.pkl` files in `offline/models/`.
  * Create a timestamped backup of the previous model weights (`model_backup_YYYYMMDD.pkl`).
  * Flush consumed entries from `buffer_daily_bars`.

> **Verification: <PHASE_5_COMPLETED>**
> 1. Execute `python -m offline.monthly_retrainer --force-run` on dummy buffer data.
> 2. Confirm new model files are output with correct timestamp backups created.
> 3. Confirm buffer records are marked as `consumed_for_training: true`.

---

### Phase 6: System Hardening & End-to-End Simulation
*Objective:* Perform end-to-end sandbox testing mimicking live trading across multiple market days.

* **Step 6.1:** Run dry-run data ingestion simulating missed days and ensure automatic data gap filling.
* **Step 6.2:** Simulate network disconnects during the online polling loop and verify automated reconnect with backoff.
* **Step 6.3:** Complete full system audit documentation and operational runbook.

> **Verification: <PHASE_6_COMPLETED>**
> 1. Full cycle test: Offline signals generated at 16:00 -> MongoDB populated -> Online worker initialized at 09:17 -> Simulated price breach -> MegaBull order fired -> Position closed -> Outcome logged in MongoDB.
> 2. Antigravity final sign-off.