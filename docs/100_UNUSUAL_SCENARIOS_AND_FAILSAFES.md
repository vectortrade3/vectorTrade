# 🛡️ VectorTrade: 100 Real-World Edge Cases & Fail-Safe Handling Catalog

This document details 100 real-world abnormal scenarios across 6 core operational domains, detailing the exact system behavior, defense mechanisms, and recovery workflows implemented across the Offline and Online systems of **VectorTrade**.

---

## 🏛️ Domain 1: Cloud & Online Micro-Worker Server Failure Scenarios (1–18)

| # | Abnormal Scenario | Root Cause | System Failure Risk | VectorTrade Defense & Recovery Mechanism |
|---|---|---|---|---|
| 1 | Online VM hard crash / power-off during market hours | Cloud host node failure | Active positions unmonitored; SL/Target missed | State is fully persisted in MongoDB `active_trades`. On automatic restart via `systemd` (`Restart=always`), worker boots in <2s, reads open trades, queries latest LTP, and resumes exit monitoring immediately. |
| 2 | Online VM Out-Of-Memory (OOM) killed | Process memory leak or memory ballooning | Abrupt termination of trading loop | Online worker has zero analytical packages (no numpy/pandas/ML), memory hard-capped at `< 50 MB` in `systemd` (`MemoryMax=60M`). If killed, `systemd` restarts within 5s and logs `OOM_RECOVERY` in `system_audit`. |
| 3 | MongoDB Atlas completely unreachable from Cloud VM | Cloud provider network partition or Atlas outage | Cannot fetch signals or update trade state | Local in-memory circuit breaker engages: worker halts any new buy triggers, uses cached memory state for existing active trades, and polls MegaBull to execute protective SL orders directly. Buffers updates in `online/pending_sync.json` until Atlas restores. |
| 4 | Offline Daemon unable to connect to Online server | Offline daemon runs on local home/office machine | Offline app cannot "push" directly to VM | **Decoupled Architecture**: Offline app *never* talks to the Online VM directly. Offline app writes strictly to MongoDB Atlas `signals_watchlist`. Cloud VM independently polls Atlas. Zero direct network dependency between Offline and Online machines. |
| 5 | Offline Daemon unable to reach MongoDB Atlas | Home ISP downtime or WiFi failure post 16:00 IST | Signals cannot be pushed to Atlas | Offline daemon saves full pre-calculated signal payload locally to `offline/pending_dispatches/signals_YYYY-MM-DD.json` and writes `DISPATCH_OFFLINE_SAVED` to `logs/eod_run_state.json`. Triggers alert for user to sync once ISP restores. |
| 6 | Cloud VM IP address rate-limited / blacklisted by MegaBull | Excessive polling frequency or IP-range block | 429 Too Many Requests or 403 Forbidden | Polling loop detects HTTP 429/403, immediately applies exponential backoff (2s $\rightarrow$ 4s $\rightarrow$ 8s up to 60s max), emits a system notification, and avoids hammering broker endpoint. |
| 7 | MegaBull API gateway returns 502/503/504 Bad Gateway / Gateway Timeout | MegaBull backend maintenance or transient cloud crash | Failed LTP fetches and order drops | `MegaBullClient` mounts `urllib3.util.retry.Retry` with automatic retry on 500, 502, 503, 504. If persistent (>3 retries), skips current tick cycle without crashing, logs warning, and retries next cycle (2.5s). |
| 8 | Cloud VM time drift (NTP desynchronization) | Virtualization clock skew | Premature expiration or delayed market start | Daemon uses `pytz.timezone('Asia/Kolkata')` and verifies UTC timestamps. Cloud startup script triggers `systemd-timesyncd` NTP force-sync before launching worker. |
| 9 | Duplicate worker instances launched simultaneously | Admin starts manual process while systemd runs | Race conditions on order dispatch (double buying) | Uses file-based mutex lock (`online/worker.lock`) on startup. Second process detects active lock, logs `CONCURRENT_INSTANCE_BLOCKED`, and exits immediately. |
| 10 | Cloud VM ephemeral disk wiped during container redeploy | Container recreate on Render/Railway | Local state loss | Online app is completely stateless locally: all armed signals, open trades, and history reside in MongoDB Atlas. Zero critical state stored on local container disk. |
| 11 | DNS resolution failure for `api.megabull.in` | Cloud provider DNS resolver drop | All HTTP calls fail with `gaierror` | Client catches `socket.gaierror` and `requests.exceptions.ConnectionError`, backs off for 5 seconds without raising unhandled crash, and logs DNS warning. |
| 12 | Incoming webhook or API endpoint flood on Dashboard | DDoS or port scan against port 8000 | Server CPU spike | FastAPI is read-only for public endpoints (`/` and `/api/state`). No external write endpoints exposed. Rate limiting headers applied. |
| 13 | Online worker started on Sunday or national market holiday | Scheduled cron triggers on holiday | Wasted CPU cycles, zero ticks | Worker checks NSE holiday calendar. If market closed or MegaBull LTP remains static/unchanged, idles in deep-sleep mode (60s sleep intervals) and disables new trade arming. |
| 14 | Worker process crashes between 09:15 and 09:17 IST | Pre-market initialization fault | Delayed startup | Guard in `execution_worker.py` loops in standby until 09:17 IST. Any crash triggers systemd automatic restart in 5s. |
| 15 | VM CPU throttled by free-tier provider (e.g. CPU credits exhausted) | Host provider CPU steal | Polling interval slows from 2.5s to 10s | Loop measures tick execution elapsed time. If elapsed time > 5s, logs `TICK_LAG_WARNING` and logs performance warning to `system_audit`. |
| 16 | Terminal session disconnects (SIGHUP sent) | User closes SSH terminal | Process killed | Service runs under systemd / supervisor daemon, detached from any TTY or interactive SSH session. |
| 17 | Cloud VM timezone configured to UTC instead of IST | Default Linux image configuration | Strategy calculates wrong market open time | Code explicitly converts all datetime comparisons to `Asia/Kolkata` using `pytz`, regardless of system host timezone setting. |
| 18 | Sudden VM shutdown during active position closing | Host eviction | Position closed at broker but not in DB | On next boot, worker reconciles with `GET /api/position/my` from MegaBull. If position is closed at broker, auto-moves local record from `active_trades` to `trade_history`. |

---

## 💻 Domain 2: Offline Daemon & Local Pipeline Exceptions (19–36)

| # | Abnormal Scenario | Root Cause | System Failure Risk | VectorTrade Defense & Recovery Mechanism |
|---|---|---|---|---|
| 19 | Offline Daemon executed at 11:00 AM on system boot | Computer turned on during market hours | Premature EOD calculations on incomplete intraday candles | **Strict Startup Time Guard**: Daemon checks `now_ist.hour < 16`. If true, logs `GUARD_TIME_NOT_MET` and terminates immediately with `sys.exit(0)`. Consumes 0% CPU, 0 MB RAM, 0% battery. |
| 20 | Offline Daemon executed multiple times in the same evening | User runs script twice or timer double-triggers | Redundant calculations, duplicate DB writes | **State & Audit Guard**: Daemon checks `logs/eod_run_state.json` and MongoDB `system_audit` for today's date (`YYYY-MM-DD`). If a successful run exists, exits immediately (`sys.exit(0)`) in <0.2 seconds. |
| 21 | Offline machine was turned off at 16:05 IST (Power outage / Laptop shut) | Missed scheduled run window | No signals generated for next morning | **Persistent Timer**: Systemd timer (`Persistent=true`) or Windows Task Scheduler (`StartWhenAvailable=true`) triggers automatically as soon as the machine powers on, evaluates time, and runs the missing EOD pipeline. |
| 22 | Offline Daemon fails to run completely (Machine remained off all night) | Power outage lasted until morning | Online worker has no fresh signals for today | **Stale Data Protection Gate in Online Worker**: Online worker checks `valid_for_date == TODAY`. If the latest signals are from yesterday or older, **online worker refuses to take any new trades**! It skips buying completely and logs `STALE_SIGNALS_REJECTED`. |
| 23 | Offline Daemon didn't run, but active positions exist from previous days | Machine off, but multi-day swing trades are open | Target or Stop Loss ignored | **Independent Multi-Day Tracking**: Online worker *always* monitors `active_trades` regardless of whether fresh signals were generated for today. It continues tracking Target and SL for open positions until closed. |
| 24 | `yfinance` download fails for a specific ticker (e.g. Yahoo 404/rate limit) | Yahoo Finance backend glitch | Pipeline crashes mid-loop | Per-symbol `try-except` block wraps data fetching. If one stock fails, logs error to `logs/offline_errors.log` and seamlessly continues processing the remaining 199 stocks. |
| 25 | Yahoo Finance returns empty DataFrame or < 50 historical bars | Newly listed or delisted stock | Indicator math fails with IndexError | Validates `len(df_bar) >= 50` before computing indicators. If invalid, skips stock cleanly. |
| 26 | Stock Symbol contains special characters (e.g. `M&M.NS`, `BAJAJ-AUTO.NS`) | URL encoding or symbol format mismatch | Download failure or token lookup mismatch | `TokenManager` sanitizes and strips hyphens and ampersands, matching both `BAJAJ-AUTO` and `BAJAJAUTO` cleanly. |
| 27 | Division by zero in technical indicator calculation (zero volume day) | Circuit breaker halt on stock | `ZeroDivisionError` or infinite indicator values | Vectorized `safe_divide(...)` with NumPy error state suppression replaces all zero divisions and NaNs with bounded floats (0.0). |
| 28 | Flat candle day (High == Low == Close == Open) | Trading suspension or illiquidity | Division by zero in `Candle_range` | `body_range_ratio`, `upper_wick_range_ratio`, and `close_position` default safely to 0.0 or 0.5 without raising exceptions. |
| 29 | NIFTY 50 benchmark download (`^NSEI`) fails | Yahoo Finance index outage | Relative strength features fail | Fallback logic populates default benchmark features (0.0 returns, 50 RSI) so single-stock indicators can still compute without crashing. |
| 30 | Missing sector classification in `stock_names.json` | Unclassified stock | Sector concentration logic breaks | Defaults sector to `"General"`. Sector filter treats unknown sectors under an isolated bucket without blocking other stocks. |
| 31 | Machine has low disk space (< 100 MB remaining) | Disk full | File write crash when writing logs or backups | Auto log rotation compresses logs over 10 MB into `.gz` archives and prunes old logs. Pending dispatches use atomic writes with disk space checks. |
| 32 | Corrupted `.pkl` model file in `models/` | Incomplete file write or filesystem corruption | Unpickling exception on daemon boot | Wrapped in `try-except`. If model load fails, logs warning to `logs/offline_errors.log`, disables that specific strategy's ML filter, and falls back to baseline technical signal parameters. |
| 33 | Model feature mismatch (trained on 99 features, data produces 98) | Column rename or indicator missing | `ValueError: Feature names mismatch` | `XGBoost` inference checks `getattr(imputer, 'feature_names_in_', None)`. Automatically realigns input DataFrame columns, filling missing features with NaN (which imputer handles cleanly). |
| 34 | Offline daemon starts while previous day's run is still hanging | Hanging network socket | Two offline runs collide | Uses single-instance process lock `offline/daemon.lock`. If locked, checks lock timestamp; if stale (>1 hour), clears lock; otherwise exits. |
| 35 | System reboot occurs during EOD Daemon execution | Power tripped at 16:02 IST mid-run | Incomplete database records | Signals are committed in atomic batches only after all calculations and risk filtering are complete. Partial runs leave no corrupted records. |
| 36 | Offline Python environment missing package after update | Accidental pip uninstall | `ModuleNotFoundError` | Entry point verifies core imports. If missing, writes critical alert to `logs/offline_errors.log` and exits cleanly with non-zero exit code. |

---

## 📉 Domain 3: Market Volatility & Trade Execution Irregularities (37–54)

| # | Abnormal Scenario | Root Cause | System Failure Risk | VectorTrade Defense & Recovery Mechanism |
|---|---|---|---|---|
| 37 | Massive Gap-Down at market open (Open < SL) | Overnight negative news | Stock triggers SL immediately without buy trigger | **Day-1 Invalidation Gate**: First tick shows $\text{LTP} \le \text{SL}$. Status transitions immediately to `INVALIDATED`. Buy order is NEVER sent. Capital 100% protected. |
| 38 | Massive Gap-Up above Buy Trigger (Open > Buy Trigger) | Overnight positive earnings | Trade entered at unfavorable high price | Online worker verifies execution type. If Open > Buy Trigger, executes at market open price, recalculates risk bracket dynamically: $\text{New Risk} = \text{Fill Price} - \text{SL}$, ensuring 1:2 R/R is preserved. |
| 39 | Extreme Volatility: Candle touches both SL and Target within 1 minute | News spike / RBI announcement | Ambiguous exit order | Worst-case protective assumption: Stop Loss takes absolute precedence. If both breached in same cycle, position is marked `SL_HIT` to preserve capital preservation discipline. |
| 40 | Upper Circuit Limit hit immediately at 09:15 IST (Stock locked at +5% / +10%) | Massive institutional buying | Buy order placed but unfilled (no sellers) | MegaBull paper trading returns order status. If order status is `PENDING` / unfilled for >10 minutes, marks `ORDER_TIMEOUT_UNFILLED` and cancels pending buy order. |
| 41 | Lower Circuit Limit hit immediately at 09:15 IST (Stock locked at -5% / -10%) | Flash crash / corporate fraud | Cannot exit long position | If position is open, order manager repeatedly queues exit sell orders. Logs `CIRCUIT_EXIT_LOCKED` warning and retains position in active monitoring until circuit opens. |
| 42 | Rapid whipsaw: Price crosses Buy Trigger by ₹0.05 then immediately crashes | False breakout / liquidity grab | Slippage and instant loss | Buy trigger requires `High + 1.0` buffer (not raw high). Once entered, strict Stop Loss at `Low - 1.0` cuts loss precisely at pre-calculated `loss_at_sl`. |
| 43 | Stock splits or bonus shares issued overnight | Corporate action | Price drops by 50% (2:1 split), falsely triggering SL | Offline daemon uses adjusted close from `yfinance`. Online worker checks for anomalous gap $>25\%$ at open; if detected, flags `CORPORATE_ACTION_SUSPECT` and halts automated trigger pending confirmation. |
| 44 | MegaBull returns stale LTP (price doesn't change for 30 minutes) | Broker quote feed freeze | Stale execution decisions | Worker compares tick timestamp or counter. If LTP is identical across 600 cycles (25 mins) during active market hours, logs `FEED_STALE_WARNING` and refrains from false triggers. |
| 45 | Negative or Zero price returned by broker (`LTP <= 0`) | Broker API corrupted payload | Division by zero or false SL trigger | Strict validation guard: `if ltp is None or ltp <= 0: continue`. Rejects invalid feed ticks completely. |
| 46 | Stock price moves rapidly past Target without filling | Slippage in fast market | Unclosed position | Target check uses inequality: `if ltp >= target_price`. Even if price jumps from 998 to 1005 (Target: 1000), target exit triggers immediately. |
| 47 | Stop Loss breached by deep wick (LTP < SL) | Liquidity wick | Unfavorable exit price | SL check uses inequality: `if ltp <= sl_price`. Exits position immediately and records exact realized P&L based on actual fill price. |
| 48 | Intraday trading halted by exchange (Technical halt) | NSE trading halt (e.g. telecom glitch) | Quotes stop updating | Polling loop detects zero quote updates, waits in standby without altering trade states, resumes as soon as quotes resume. |
| 49 | Signal generated on stock with extremely low liquidity | Low volume constituent | High bid-ask spread | Indicator pipeline calculates `volume_avg_20d`. Risk manager filters out any symbol with 20-day average volume below minimum liquidity threshold ($< 50,000$ shares/day). |
| 50 | Sudden exchange-wide crash (NIFTY drops 4% in 15 minutes) | Black swan event | Multiple SLs hit simultaneously | Cross-portfolio circuit breaker: if $>3$ positions hit SL in same session, halts all remaining `ARMED` signals and preserves remaining cash. |
| 51 | Target reached on Day 7 at 15:29 IST | Last minute target fill | Position holding window expiration | Position tracker monitors `days_active`. If day 7 reaches 15:30 IST without Target or SL hit, position is closed under `TIMEOUT_EXIT`. |
| 52 | SL and Target are within ₹0.50 of each other (Extremely tight range) | Inappropriate candle structure | Micro risk, false trigger | Risk manager requires minimum risk distance: $\text{Buy} - \text{SL} \ge 0.5\% \text{ of price}$. Rejects unnaturally tight noise setups. |
| 53 | Stop Loss price calculated is negative (Stock price < ₹1.0) | Penny stock anomaly | Invalid order parameters | Guard checks `sl_price > 0`. Discards any setup where calculated SL is $\le 0$. |
| 54 | High order slippage reported by MegaBull | Simulated execution lag | PnL variance from plan | Trade history calculates and records `fill_price` vs `buy_trigger` slippage, saving `slippage_cost` to MongoDB for audit. |

---

## 💰 Domain 4: Capital, Position Sizing & Margin Constraints (55–70)

| # | Abnormal Scenario | Root Cause | System Failure Risk | VectorTrade Defense & Recovery Mechanism |
|---|---|---|---|---|
| 55 | Risk per share is very small, resulting in huge calculated Quantity | Low volatility consolidation | Position exceeds available account balance | Risk manager caps maximum capital per trade: $\text{Quantity} = \min(\lfloor \text{Risk} / \text{Risk\_Share} \rfloor, \lfloor \text{Max\_Capital\_Per\_Trade} / \text{Buy} \rfloor)$. |
| 56 | Account balance on MegaBull is lower than required capital | Prior paper losses reduced virtual cash | Broker rejects order with `INSUFFICIENT_FUNDS` | Worker queries `/api/user/my` for available cash before firing order. If insufficient, scales down quantity or skips trade with `INSUFFICIENT_MARGIN` audit log. |
| 57 | All 5 slots filled, 6th high-conviction signal triggers | Active market breakout day | Margin over-allocation | `max_open_positions = 5` strictly enforced by `PortfolioRiskManager` offline. Online worker double-checks open positions count before dispatch. |
| 58 | 4 out of 5 approved signals belong to the IT sector | Sector-wide rally | Excessive sector concentration risk | `PortfolioRiskManager` enforces maximum 30% allocation per sector (max 1–2 trades per sector). Extra IT signals rejected before reaching database. |
| 59 | Both Strategy 1 and Strategy 2 issue BUY on same stock | Both models detect setup | Duplicate buy orders, double exposure | Correlation Deduplication Gate: Retains only the signal with the higher ML confidence score; drops duplicate. |
| 60 | User manually closes a position on MegaBull web UI | Manual human intervention | State mismatch between MegaBull and MongoDB | Online worker reconciles open positions with `GET /api/position/my`. If position is missing on broker, moves local trade to `trade_history` with status `MANUAL_EXTERNAL_EXIT`. |
| 61 | User manually buys a stock on MegaBull web UI | Discretionary manual trade | System monitors unmanaged trade | Worker only monitors positions containing `VectorTrade` order tags or matching `active_trades` MongoDB IDs. Ignores manual user trades. |
| 62 | Virtual account balance reset to ₹0 by user on web portal | Account reset | All subsequent paper orders fail | Worker catches broker rejection message `"Insufficient Balance"`, disables new entries, alerts user via dashboard notification badge. |
| 63 | Cumulative portfolio drawdown exceeds 10% in a month | Extended market drawdown | Deep portfolio loss | Retrainer and Risk Manager trigger `DRAWDOWN_DEFENSE_MODE`: halves `risk_per_trade_pct` from 1.5% to 0.75% until recovery. |
| 64 | Fractional share calculated (`Quantity = 4.7`) | Non-integer division | Broker rejects non-integer share quantity | `Quantity` is strictly cast to integer using `int(floor(...))`. Minimum quantity clamped to `1`. |
| 65 | Total calculated position cost is ₹0 | Quantity calculated as 0 | Broker rejects 0 qty order | Any signal resulting in `quantity <= 0` is discarded during the offline risk phase. |
| 66 | Total portfolio risk limit breached by existing open trades | Gap-down enlarged risk | Uncontrolled portfolio loss | Circuit Breaker: checks cumulative open risk $\sum (\text{Buy} - \text{SL}) \times \text{Qty}$. If $> 7.5\%$ of capital, completely blocks new signal arming. |
| 67 | Multiple strategies trigger simultaneously at 09:18 IST | Opening bell breakout | Broker API flooded with concurrent orders | Worker processes triggers sequentially with 200ms spacing between HTTP order dispatches to prevent rate-limit throttling. |
| 68 | MegaBull rejects order with `"Invalid Lot Size"` | Index or F&O token treated as equity | Order failure | Universe restricted strictly to NSE Equity segment (`CNC` / Delivery). Derivatives excluded from equity pipeline. |
| 69 | Target price reaches astronomical value due to formula glitch | Data spike | Target never reached | Target price sanity check: validates $\text{Target} \le \text{Buy} \times 1.5$. Flags warning if target exceeds 50% gain in 7 days. |
| 70 | Stop loss set higher than Buy Trigger | Math calculation error | Instant exit / order rejection | Hard assertion: `assert buy_trigger > sl_price`. Discards corrupt setup immediately. |

---

## 🗄️ Domain 5: Database, Caching & Data Consistency Scenarios (71–86)

| # | Abnormal Scenario | Root Cause | System Failure Risk | VectorTrade Defense & Recovery Mechanism |
|---|---|---|---|---|
| 71 | MongoDB Atlas primary replica fails over | Atlas automated maintenance | 10–30s connection drop | `DatabaseManager` wraps queries in reconnection retry loop with `AutoReconnect` handling. Reconnects seamlessly to new primary. |
| 72 | MongoDB Atlas storage quota exceeded (Free tier 512 MB full) | Unbounded logging or historical collections | Database writes rejected (`StorageFull`) | `buffer_daily_bars` is consumed and flushed monthly. Heavy 5-year data remains in local CSV/parquet, NOT dumped into Atlas free tier. |
| 73 | `signals_watchlist` contains old signals from 3 days ago | Offline daemon didn't run recently | Online worker trades old outdated signals | Worker enforces `valid_for_date == TODAY`. Any signal where `valid_for_date != TODAY` is completely ignored for new entries. |
| 74 | `tokens_cache.json` deleted or missing | Accidental file deletion | Token lookup failure | `TokenManager` detects missing cache, automatically downloads fresh CSV from MegaBull AWS S3, and rebuilds cache in <3 seconds. |
| 75 | MegaBull modifies instrument token for a stock | Exchange token reassignment | Orders placed on wrong instrument | Tokens refreshed automatically during initial setup and refreshed whenever a token lookup returns 404 or `"Script not found"`. |
| 76 | Corrupted document inserted in MongoDB (missing `sl_price`) | Incomplete write | Python `KeyError` crashes online worker | Worker uses safe dictionary lookups: `float(sig.get('sl_price', 0.0))`. Skips invalid records and logs schema error to `system_audit`. |
| 77 | Network disconnect occurs in the exact millisecond after order sent but before DB update | Power / network glitch | Order executed at broker but not in MongoDB | On reconnect, worker reconciles with `GET /api/order/my`. If an executed order exists without matching DB trade, auto-creates `active_trades` entry. |
| 78 | Multiple reads/writes to `eod_run_state.json` simultaneously | Concurrent processes | JSONDecodeError / corrupt file | File writes use atomic rename strategy: writes to `eod_run_state.json.tmp` then atomically replaces target file via `os.replace`. |
| 79 | Database query times out (> 10s) | Heavy network congestion | Thread hangs indefinitely | All PyMongo operations configure `socketTimeoutMS=10000` and `serverSelectionTimeoutMS=10000`. Times out cleanly instead of hanging. |
| 80 | Duplicate document ID collision in MongoDB `_id` | Rerun on same date | `DuplicateKeyError` crashes script | All writes use `update_one({"_id": ...}, {"$set": ...}, upsert=True)`. Idempotent updates prevent duplicate key crashes. |
| 81 | `trade_history` collection corrupted | Data corruption | Historical analytics unavailable | In-memory trade history backup maintained in `logs/trade_history_backup.jsonl` (appends every closed trade locally). |
| 82 | MongoDB Atlas credentials in `.env` expired or rotated | Password change | Authentication failed (401) | `core/db.py` catches `OperationFailure`, logs clear message `"Check MONGODB_URI credentials in .env"`, and falls back to local in-memory mock during test runs. |
| 83 | MongoDB Atlas IP Access List blocked (Dynamic IP changed) | ISP changed home router IP | Connection refused | Logs clear warning: `"IP not whitelisted on MongoDB Atlas"`. Offline daemon saves to `offline/pending_dispatches/` so signals are not lost. |
| 84 | System audit collection grows excessively large | Daily audit records over years | Excessive document count | Indexed by `timestamp: -1`. Queries use `.limit(50)`. Retention policy auto-prunes audit logs older than 90 days. |
| 85 | Stale trade left in `active_trades` for 30 days | Broker failed to report exit | Slot permanently occupied | Position tracker checks `entry_timestamp`. If trade is open for $> 10$ trading days, flags `OVERDUE_TRADE_FORCE_AUDIT` and triggers reconciliation. |
| 86 | Offline daemon saves signals to `pending_dispatches/`, then runs again next day | Accumulated pending files | Multiple days of pending dispatches | Sync engine processes pending dispatches chronologically, syncing only current-day valid signals to Atlas and archiving older ones. |

---

## 🔒 Domain 6: Security, User Notification & Broker API Edge Cases (87–100)

| # | Abnormal Scenario | Root Cause | System Failure Risk | VectorTrade Defense & Recovery Mechanism |
|---|---|---|---|---|
| 87 | MegaBull API Key expires (1-month validity window expires) | 30-day token expiration | All API calls return 401 Unauthorized | **Instant User Notification Badge**: Online worker detects HTTP 401, sets status flag `BROKER_AUTH_EXPIRED`, displays red alert banner across top of Dashboard UI, and halts further order attempts until user updates `.env`. |
| 88 | Accidental commit of `.env` file containing secret keys | Developer git error | Credential leak on GitHub | `.gitignore` strictly excludes `.env`, `.env.*` (except `.env.example`). Git hooks and CI checks audit repository for credential patterns before commit. |
| 89 | MegaBull API returns HTML error page instead of JSON | Cloudflare / Nginx 502 page | `json.decoder.JSONDecodeError` | Client verifies `response.headers.get('content-type')`. If not `application/json`, logs raw status code and avoids crashing JSON parser. |
| 90 | MegaBull changes response schema (e.g. `ltp` renamed to `lastPrice`) | Upstream API update | NoneType / Key error | Multi-key resolution: `data.get('ltp') or data.get('lastPrice') or data.get('price')`. If all fail, logs schema evolution warning. |
| 91 | Unauthorized access attempt to Online Dashboard | Port exposed to public internet | Malicious manipulation | Dashboard is strictly read-only. Optional HTTP Basic Auth / API token header configured via `DASHBOARD_SECRET_KEY` in `settings.py`. |
| 92 | MegaBull account locked due to excessive invalid logins | Security lock | Service outage | Emits urgent alert in `system_audit` and displays `ACCOUNT_LOCKED` banner on dashboard. |
| 93 | Broker returns Order ID as string in one endpoint, int in another | Inconsistent API types | Type mismatch in DB queries | Strict normalization: all order IDs and instrument tokens are cast via `str(...)` across all storage and comparison points. |
| 94 | Cross-Site Scripting (XSS) attempt via Stock Symbol | Malicious symbol payload in DB | Dashboard compromised | Dashboard uses safe Jinja2 templating with automatic HTML entity escaping enabled (`autoescape=True`). |
| 95 | MegaBull order endpoint experiences 15-second latency | Broker database locking | Online worker loop blocked | All HTTP calls enforce `timeout=10.0`. Worker executes in background thread so dashboard remains responsive. |
| 96 | User changes `.env` while online daemon is running | Live credential update | Stale running configuration | Online dashboard provides `/api/reload-config` endpoint and file watcher to reload settings without process restart. |
| 97 | Python process receives SIGTERM from Docker/Kubernetes | Container stop/restart | Ungraceful termination mid-cycle | Signal handler catches `SIGTERM` / `SIGINT`, closes open HTTP sessions, flushes in-memory audit logs, and exits cleanly within 2 seconds. |
| 98 | Network latency spikes from 50ms to 2500ms | 4G/5G tethering or congested wifi | Missed breakout ticks | Worker logs latency metrics. If tick latency consistently exceeds 2s, increases polling interval to prevent HTTP request backlog. |
| 99 | MegaBull server time differs from Indian Standard Time | Broker server NTP error | Conflicting trigger times | System relies on authoritative local system time for all market window decisions (09:17 – 15:32 IST), never broker server headers. |
| 100 | Disaster Recovery: Total Cloud and Local failure simultaneously | Simultaneous outages | Unknown system state | Complete immutable audit trail maintained in MongoDB `trade_history` and `system_audit`. System can be completely reconstructed from MongoDB logs on any fresh machine in under 3 minutes. |

---

## 🎯 Summary of Key Architectural Guardrails Implemented:

1. **Decoupled Asynchrony**: Offline never depends on Online being reachable. They coordinate strictly via MongoDB Atlas.
2. **Stale Signal Elimination**: Online worker validates `valid_for_date == TODAY`. If Offline daemon didn't run, Online will **never** trade on old setups.
3. **Continuous Open Trade Protection**: Online worker tracks existing open trades regardless of whether new signals were generated today.
4. **Day-1 Invalidation Gate**: If price touches SL before Buy Trigger, the signal is dropped permanently—zero false orders.
5. **Zero-Idle Headless Execution**: Offline script terminates with `sys.exit(0)` before 16:00 IST or after completion, consuming 0 MB RAM and 0% CPU.
6. **Universal Alerting**: Expired MegaBull keys, database drops, and invalid states surface live on the dark-themed real-time dashboard.
