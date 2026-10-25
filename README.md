# NIFTY 200 ML Trading Strategy Pipeline

This project builds a Machine Learning layer on top of a base algorithmic trading strategy (50MA + Bollinger Band pullbacks). The goal is to filter out false signals from the base strategy using an ML classification model trained on 130+ technical and contextual features.

## 🗂 Project Structure & Scripts

### 1. Data Downloading
* **Script:** `download_historical_data.py`
* **What it does:** Fetches historical data (OHLCV) for the NIFTY 200 constituent stocks via `yfinance`. It pulls roughly 900 calendar days and trims it precisely to the last **565 working days** up to yesterday (excluding today's live data). It also queries Yahoo Finance to dynamically retrieve the `Sector` for each stock.
* **Output:** Saves 200 individual stock CSV files into the `stock_data/` folder.

### 2. Feature Generation & Trade Simulation
* **Notebook:** `feature_generation.ipynb`
* **What it does:** Reads the raw data and calculates 100+ technical features:
  * Moving Averages, RSI, ATR, OBV.
  * Bollinger Bands and Distances.
  * Detailed Candlestick structural features (Wicks, Dojis, Engulfing).
  * Evaluates the **Base Strategy Signal**: 50MA trend up + Price near Lower BB + Green Candle.
  * **Chronological Simulator:** Performs a strictly chronological Day 1 to Day 7 forward walk to accurately determine if the Target (1:2 Risk/Reward) or Stop Loss was hit first. It specifically accounts for intra-day ambiguity on Day 1 (Trigger logic).
* **Output:** Saves 200 highly enriched CSV files into the `data_generated/` folder.

### 3. Combining & Formatting
* **Notebook:** `combine_data.ipynb`
* **What it does:** Consolidates all 200 enriched files into a single massive dataframe (110,000+ rows). It maps all column names to a strict `snake_case` format, drops any unnecessary raw columns, and calculates cross-sectional metrics (like Sector returns and cross-over durations).
* **Output:** Saves the consolidated master dataset to `data_final.csv` in the root directory.

### 4. Phase 1: Data Validation
* **Script:** `phase1_validate.py`
* **What it does:** Scans `data_final.csv` for data integrity. It checks for:
  * Duplicates (0 found).
  * Missing values (expected due to rolling metrics).
  * Infinite values (Found ~980 division-by-zero errors in volume change, to be handled in Phase 5).
  * Negative prices/volumes.
* **Output:** Sorts the master CSV chronologically and saves a detailed report to `validation_report.txt`.

### 5. Phase 2 & 3: Signal Dataset Extraction
* **Script:** `prepare_ml_dataset.py`
* **What it does:** 
  * Filters the 110,000+ rows down to **only the days where a trade actually triggered** (`base_signal == True` AND trade executed on Day 1).
  * Removes "Pending" signals (last 7 days of dataset) and "Timeouts" (trades that didn't hit Target/SL within 7 days).
  * Converts the text Outcome into our ML Answer Key: `ml_target` (1 for Target, 0 for SL).
* **Output:** Saves the 1,825 valid trades to `ml_dataset_phase3.csv`.

### 6. Phase 4 onwards: Machine Learning Pipeline
* **Notebook:** `trade_final_analysis.ipynb` *(Main working notebook)*
* **What it does:** 
  * Loads `ml_dataset_phase3.csv`.
  * **Phase 4 (Data Leakage Prevention):** Strictly drops any column containing future information (e.g., `max_gain_7d`, `outcome`, `target_hit`) to prevent the model from cheating.
  * Saves the pristine, leak-free training dataset to `final_trade_data.csv`.
  * *Future phases (Phase 5+, feature selection, modeling) will continue in this notebook.*
* **Output:** `final_trade_data.csv` (The final dataset used for all ML training).

## 📂 Data Storage & File Tracking

### 1. `stock_data/` (Folder)
* **What it contains:** 200 individual CSV files (one for each NIFTY 200 stock).
* **Record Type:** Raw historical market data.
* **Columns Included:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `Sector`.

### 2. `data_generated/` (Folder)
* **What it contains:** 200 individual CSV files.
* **Record Type:** Enriched historical data with calculated technical features and simulated target outcomes.
* **Columns Included:** The raw OHLCV + 130 technical indicators (Moving Averages, RSI, Bollinger Bands, ATR, Candle characteristics, Sector metrics) + Simulated Target/SL logic variables (`Target_Hit`, `SL_Hit`, `Days_to_Outcome`, `MFE`, `MAE`).

### 3. `data_final.csv`
* **What it contains:** A massive single spreadsheet (110,000+ rows) combining all files from `data_generated/`.
* **Record Type:** Universal dataset of all trading days across all stocks.
* **Columns Included:** Contains exactly 156 standardized columns (all renamed to strict `snake_case`), covering OHLCV, moving averages, ratios, slopes, RSIs, Bollinger details, gap percentages, and target outcomes. *This dataset contains data leakage columns.*

### 4. `ml_dataset_phase3.csv`
* **What it contains:** A filtered subset containing exactly **1,825 rows**.
* **Record Type:** Only the days where `base_signal == True` and the trade actually triggered on Day 1. It excludes "Pending" and "Timeout" trades.
* **Columns Included:** All 156 columns from `data_final.csv`, plus `ml_target` (1 for Target hit, 0 for Stop Loss hit). *This dataset still contains data leakage columns for reference.*

### 5. `final_trade_data.csv`
* **What it contains:** The exact same 1,825 rows as `ml_dataset_phase3.csv`.
* **Record Type:** A clean, strictly non-leaking ML training dataset.
* **Columns Included:** The `ml_target` and all past technical features. **Removed:** Any column containing future knowledge (`target_hit`, `outcome`, `max_gain_7d`, `mfe`, etc.).

### 6. `validation_report.txt`
* **What it contains:** A text report verifying the integrity of `data_final.csv`.
* **Record Type:** Diagnostic text (checks for duplicates, nulls, constants, infinite values).
