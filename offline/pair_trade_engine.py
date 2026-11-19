"""
Institutional Cointegrated Pairs Trading Strategy v4 - Trade Engine
Pipeline Module: offline/pair_trade_engine.py

Performs:
1. Aligned daily series loading for pairs from curated_pairs.json
2. Rolling 90-day OLS hedge ratio (beta) and rolling spread calculation
3. Spread Z-Score and momentum velocity slope (Delta Z)
4. Full technical indicator calculation for Stock A and Stock B (SMA 20/50/200, RSI 14, ATR 14, HistVol 20, VolRatio 20)
5. Pair level statistics: Rolling 90D correlation, beta, ADF t-stat, Ornstein-Uhlenbeck half-life, spread volatility %
6. Config-driven trade entry filtering: 2.10 <= |Z| <= 3.00, |Delta Z| <= 0.05
7. Next 7-day forward horizon tracking (T+1 to T+7 daily prices, spreads, and z-scores for both legs)
8. Trade simulation until Target (|Z| <= 0.40), Stop Barrier (|Z| >= 2.85), or Time Stop (holding cap)
9. Outcome labeling: Success (1), Fail (0), Neutral (2) and average holding time tracking
10. Strict 4-year training (<= 2025-11-18) vs 1-year holdout testing (> 2025-11-18) dataset serialization
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PairTradeEngine")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_STORE_DIR = BASE_DIR / "data_store"
RAW_DATA_DIR = BASE_DIR / "5_year_data" / "raw"
PAIR_STRATEGY_DIR = DATA_STORE_DIR / "pair_strategy"
CONFIG_FILE = BASE_DIR / "pair_strategy_v4_config.json"
CURATED_PAIRS_FILE = DATA_STORE_DIR / "curated_pairs.json"


def load_strategy_config() -> Dict[str, Any]:
    """Loads institutional configuration for pair strategy v4."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes standard Relative Strength Index (RSI) using exponential Wilder smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.bfill().fillna(50.0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Computes Average True Range as percentage of close."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    atr_pct = (atr / (close + 1e-9)) * 100.0
    return atr_pct.fillna(2.0)


def compute_half_life(spread_series: pd.Series) -> float:
    """
    Computes Ornstein-Uhlenbeck mean-reversion half-life tau_HL:
    delta_y_t = theta * y_{t-1} + const + eps
    tau_HL = -ln(2) / theta
    """
    try:
        s = spread_series.dropna()
        if len(s) < 20:
            return 10.0
        lagged = s.shift(1).dropna()
        delta = s.diff().dropna()
        common_idx = lagged.index.intersection(delta.index)
        x = lagged.loc[common_idx].values
        y = delta.loc[common_idx].values
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-12:
            return 10.0
        theta = np.sum((x - x_mean) * (y - y_mean)) / var_x
        if theta >= 0:
            return 30.0  # Non-mean-reverting, clamp high
        hl = -np.log(2) / theta
        return float(np.clip(hl, 1.0, 45.0))
    except Exception:
        return 10.0


def compute_adf_stat(spread_series: pd.Series) -> float:
    """Quick proxy for ADF test t-statistic using linear regression."""
    try:
        s = spread_series.dropna()
        if len(s) < 20:
            return -2.0
        lagged = s.shift(1).dropna()
        delta = s.diff().dropna()
        common_idx = lagged.index.intersection(delta.index)
        x = lagged.loc[common_idx].values
        y = delta.loc[common_idx].values
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-12:
            return 0.0
        beta = np.sum((x - x_mean) * (y - y_mean)) / var_x
        alpha = y_mean - beta * x_mean
        residuals = y - (alpha + beta * x)
        s_err = np.sqrt(np.sum(residuals ** 2) / max(1, n - 2))
        se_beta = s_err / np.sqrt(var_x)
        t_stat = beta / (se_beta + 1e-9)
        return float(t_stat)
    except Exception:
        return -2.0


class PairTradeEngine:
    """Engine responsible for extracting, enriching, labeling, and persisting pair trades."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_strategy_config()
        self.lookback_days = int(self.config.get("divergence_triggers", {}).get("lookback_days", 90))
        self.z_entry_hurdle = float(self.config.get("divergence_triggers", {}).get("z_score_entry_hurdle", 2.10))
        self.z_entry_max = float(self.config.get("divergence_triggers", {}).get("z_score_entry_max", 3.00))
        self.max_momentum_slope = float(self.config.get("divergence_triggers", {}).get("max_entry_momentum_slope", 0.05))
        self.z_exit_target = float(self.config.get("divergence_triggers", {}).get("z_score_exit_target", 0.40))
        self.z_stop_loss = float(self.config.get("divergence_triggers", {}).get("z_score_stop_loss_barrier", 2.85))
        
        time_stops = self.config.get("time_and_event_stops", {})
        self.hl_stop_mult = float(time_stops.get("half_life_time_stop_multiplier", 2.5))
        self.min_holding_days = int(time_stops.get("min_holding_days_floor", 4))
        self.max_holding_days = int(time_stops.get("max_holding_days_cap", 22))

        # Split cutoff date: 4 years train (<= 2025-11-18), 1 year holdout test (> 2025-11-18)
        self.split_cutoff_date = "2025-11-18"

    def load_stock_df(self, ticker: str) -> Optional[pd.DataFrame]:
        """Loads and cleans raw stock CSV."""
        path = RAW_DATA_DIR / f"{ticker}_data.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if "Date" not in df.columns or "Close" not in df.columns:
                return None
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")
            return None

    def enrich_single_stock_features(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Calculates single stock technical indicators."""
        df = df.copy()
        close = df["Close"]
        high = df["High"] if "High" in df.columns else close
        low = df["Low"] if "Low" in df.columns else close
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)

        # SMAs & Ratios
        sma20 = close.rolling(20, min_periods=10).mean()
        sma50 = close.rolling(50, min_periods=20).mean()
        sma200 = close.rolling(200, min_periods=50).mean()

        df[f"{prefix}_SMA_20_Ratio"] = (close / sma20).fillna(1.0)
        df[f"{prefix}_SMA_50_Ratio"] = (close / sma50).fillna(1.0)
        df[f"{prefix}_SMA_200_Ratio"] = (close / sma200).fillna(1.0)

        # RSI & ATR
        df[f"{prefix}_RSI_14"] = compute_rsi(close, 14)
        df[f"{prefix}_ATR_14_Pct"] = compute_atr(high, low, close, 14)

        # 20-Day Annualized Historical Volatility
        returns = close.pct_change()
        df[f"{prefix}_HistVol_20"] = (returns.rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0).fillna(20.0)

        # Volume Ratio
        vol_sma20 = volume.rolling(20, min_periods=5).mean()
        df[f"{prefix}_Volume_Ratio_20"] = (volume / (vol_sma20 + 1e-9)).fillna(1.0)

        return df

    def compute_pair_time_series(self, ticker_a: str, ticker_b: str) -> Optional[pd.DataFrame]:
        """Aligns time series of both stocks and calculates rolling OLS beta, spread, and Z-score."""
        df_a = self.load_stock_df(ticker_a)
        df_b = self.load_stock_df(ticker_b)
        if df_a is None or df_b is None:
            return None

        # Add single-stock features
        df_a = self.enrich_single_stock_features(df_a, "A")
        df_b = self.enrich_single_stock_features(df_b, "B")

        # Merge on Date
        cols_a = ["Date", "Close", "A_SMA_20_Ratio", "A_SMA_50_Ratio", "A_SMA_200_Ratio", "A_RSI_14", "A_ATR_14_Pct", "A_HistVol_20", "A_Volume_Ratio_20"]
        cols_b = ["Date", "Close", "B_SMA_20_Ratio", "B_SMA_50_Ratio", "B_SMA_200_Ratio", "B_RSI_14", "B_ATR_14_Pct", "B_HistVol_20", "B_Volume_Ratio_20"]

        merged = pd.merge(
            df_a[cols_a].rename(columns={"Close": "Close_A"}),
            df_b[cols_b].rename(columns={"Close": "Close_B"}),
            on="Date",
            how="inner"
        ).sort_values("Date").reset_index(drop=True)

        if len(merged) < self.lookback_days + 15:
            return None

        merged["Log_A"] = np.log(merged["Close_A"])
        merged["Log_B"] = np.log(merged["Close_B"])

        # Rolling OLS Hedge Ratio (beta) over lookback_days
        roll_cov = merged["Log_A"].rolling(self.lookback_days).cov(merged["Log_B"])
        roll_var = merged["Log_B"].rolling(self.lookback_days).var()
        merged["Hedge_Ratio_Beta"] = (roll_cov / (roll_var + 1e-9)).bfill().fillna(1.0)
        merged["Hedge_Ratio_Beta"] = merged["Hedge_Ratio_Beta"].clip(lower=0.05, upper=10.0)

        # Rolling Spread: ln(P_A) - beta * ln(P_B)
        merged["Spread"] = merged["Log_A"] - (merged["Hedge_Ratio_Beta"] * merged["Log_B"])

        # Rolling Spread Mean & Std
        spread_mean = merged["Spread"].rolling(self.lookback_days).mean()
        spread_std = merged["Spread"].rolling(self.lookback_days).std()

        # Spread Z-Score
        merged["Z_Score"] = ((merged["Spread"] - spread_mean) / (spread_std + 1e-9)).fillna(0.0)

        # Momentum velocity slope: Delta Z = Z(t) - Z(t-1)
        merged["Delta_Z"] = merged["Z_Score"].diff().fillna(0.0)

        # Pair level cross-features
        merged["Rolling_Corr_90"] = merged["Log_A"].rolling(self.lookback_days).corr(merged["Log_B"]).fillna(0.85)
        merged["Spread_Vol_Pct_90"] = (spread_std * 100.0).fillna(3.5)
        merged["Vol_Ratio_A_to_B"] = (merged["A_HistVol_20"] / (merged["B_HistVol_20"] + 1e-9)).fillna(1.0)
        merged["RSI_Diff_A_minus_B"] = merged["A_RSI_14"] - merged["B_RSI_14"]

        return merged

    def simulate_trade_from_entry(
        self,
        pair_df: pd.DataFrame,
        entry_idx: int,
        pair_meta: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Simulates trade forward from entry until target, barrier stop, or half-life time cap."""
        n_bars = len(pair_df)
        if entry_idx >= n_bars - 8:
            return None  # Needs at least 7 forward bars

        entry_row = pair_df.iloc[entry_idx]
        entry_date = entry_row["Date"]
        z_entry = float(entry_row["Z_Score"])
        delta_z = float(entry_row["Delta_Z"])

        # Check entry criteria
        # Long Spread: z in [-3.00, -2.10], Short Spread: z in [2.10, 3.00]
        is_long = (-self.z_entry_max <= z_entry <= -self.z_entry_hurdle)
        is_short = (self.z_entry_hurdle <= z_entry <= self.z_entry_max)

        if not (is_long or is_short):
            return None

        # Divergence momentum flattening confirmation
        if abs(delta_z) > self.max_momentum_slope:
            return None

        trade_dir = 1 if is_long else -1  # 1: Long Spread (Buy A, Short B); -1: Short Spread (Short A, Buy B)
        p_a_entry = float(entry_row["Close_A"])
        p_b_entry = float(entry_row["Close_B"])
        beta_entry = float(entry_row["Hedge_Ratio_Beta"])

        # Compute dynamic Ornstein-Uhlenbeck half-life over recent spread lookback
        spread_window = pair_df["Spread"].iloc[max(0, entry_idx - self.lookback_days + 1): entry_idx + 1]
        half_life = compute_half_life(spread_window)
        adf_stat = compute_adf_stat(spread_window)
        max_holding_limit = int(np.clip(int(np.floor(self.hl_stop_mult * half_life)), self.min_holding_days, self.max_holding_days))

        # Forward tracking: Next 7 days exact trajectory
        forward_7d = []
        for f_step in range(1, 8):
            f_idx = entry_idx + f_step
            if f_idx < n_bars:
                f_row = pair_df.iloc[f_idx]
                f_pa = float(f_row["Close_A"])
                f_pb = float(f_row["Close_B"])
                f_ret = trade_dir * (((f_pa - p_a_entry) / p_a_entry) - beta_entry * ((f_pb - p_b_entry) / p_b_entry)) * 100.0
                forward_7d.append({
                    "step": f_step,
                    "date": f_row["Date"],
                    "price_a": f_pa,
                    "price_b": f_pb,
                    "spread": float(f_row["Spread"]),
                    "zscore": float(f_row["Z_Score"]),
                    "pair_return_pct": round(f_ret, 3)
                })

        # Simulate until terminal exit
        exit_idx = None
        exit_reason = "HOLDING_CAP"
        mfe_pct = -999.0
        mae_pct = 999.0

        for cur_idx in range(entry_idx + 1, min(n_bars, entry_idx + max_holding_limit + 1)):
            cur_row = pair_df.iloc[cur_idx]
            cur_pa = float(cur_row["Close_A"])
            cur_pb = float(cur_row["Close_B"])
            cur_z = float(cur_row["Z_Score"])

            cur_ret = trade_dir * (((cur_pa - p_a_entry) / p_a_entry) - beta_entry * ((cur_pb - p_b_entry) / p_b_entry)) * 100.0
            if cur_ret > mfe_pct:
                mfe_pct = cur_ret
            if cur_ret < mae_pct:
                mae_pct = cur_ret

            # Check 1: Target exit (|Z| <= 0.40)
            if abs(cur_z) <= self.z_exit_target:
                exit_idx = cur_idx
                exit_reason = "TARGET_REACHED"
                break

            # Check 2: Dynamic stop-loss barrier (|Z| >= 2.85)
            if abs(cur_z) >= self.z_stop_loss:
                # Barrier breached against position
                if (trade_dir == 1 and cur_z < -self.z_stop_loss) or (trade_dir == -1 and cur_z > self.z_stop_loss):
                    exit_idx = cur_idx
                    exit_reason = "STOP_LOSS_BARRIER"
                    break

        if exit_idx is None:
            if entry_idx + max_holding_limit < n_bars:
                exit_idx = entry_idx + max_holding_limit
                exit_reason = "TIME_STOP_CAP"
            else:
                exit_idx = n_bars - 1
                exit_reason = "DATA_END_TRUNCATION"

        exit_row = pair_df.iloc[exit_idx]
        exit_date = exit_row["Date"]
        p_a_exit = float(exit_row["Close_A"])
        p_b_exit = float(exit_row["Close_B"])
        z_exit = float(exit_row["Z_Score"])
        holding_days = exit_idx - entry_idx

        # Calculate Net Return
        net_return_pct = trade_dir * (((p_a_exit - p_a_entry) / p_a_entry) - beta_entry * ((p_b_exit - p_b_entry) / p_b_entry)) * 100.0

        # Outcome Labeling strictly adhering to Phase 2:
        # If STOP_LOSS_BARRIER is triggered, the divergence broke out -> FAIL (0)
        # If TARGET_REACHED or positive return >= +1.50% without hitting stop barrier -> SUCCESS (1)
        # If STOP_LOSS_BARRIER or negative return < -1.00% -> FAIL (0)
        # Otherwise (time stop with -1.00% <= return < +1.50%) -> NEUTRAL (2)
        if exit_reason == "STOP_LOSS_BARRIER":
            outcome_label = 0
            outcome_text = "FAIL"
        elif exit_reason == "TARGET_REACHED" or net_return_pct >= 1.50:
            outcome_label = 1
            outcome_text = "SUCCESS"
        elif net_return_pct < -1.00:
            outcome_label = 0
            outcome_text = "FAIL"
        else:
            outcome_label = 2
            outcome_text = "NEUTRAL"

        # Determine split (Train <= 2025-11-18, Test > 2025-11-18)
        split_type = "TRAIN" if entry_date <= self.split_cutoff_date else "TEST"

        trade_record = {
            "pair_id": pair_meta.get("id", f"{pair_meta['ticker_a']}_{pair_meta['ticker_b']}"),
            "ticker_a": pair_meta["ticker_a"],
            "ticker_b": pair_meta["ticker_b"],
            "sector": pair_meta.get("sector", "Unknown"),
            "trade_direction": "LONG_SPREAD" if trade_dir == 1 else "SHORT_SPREAD",
            "entry_date": entry_date,
            "exit_date": exit_date,
            "holding_days": holding_days,
            "max_holding_limit": max_holding_limit,
            "exit_reason": exit_reason,
            "p_a_entry": p_a_entry,
            "p_b_entry": p_b_entry,
            "p_a_exit": p_a_exit,
            "p_b_exit": p_b_exit,
            "hedge_ratio_beta": round(beta_entry, 4),
            "z_score_entry": round(z_entry, 3),
            "delta_z_entry": round(delta_z, 4),
            "z_score_exit": round(z_exit, 3),
            "net_return_pct": round(net_return_pct, 3),
            "mfe_pct": round(mfe_pct if mfe_pct != -999.0 else 0.0, 3),
            "mae_pct": round(mae_pct if mae_pct != 999.0 else 0.0, 3),
            "half_life_days": round(half_life, 2),
            "adf_stat": round(adf_stat, 2),
            "outcome_label": outcome_label,
            "outcome_text": outcome_text,
            "split_type": split_type,
            # Dual-stock technical features at entry
            "a_sma_20_ratio": round(float(entry_row["A_SMA_20_Ratio"]), 4),
            "a_sma_50_ratio": round(float(entry_row["A_SMA_50_Ratio"]), 4),
            "a_sma_200_ratio": round(float(entry_row["A_SMA_200_Ratio"]), 4),
            "a_rsi_14": round(float(entry_row["A_RSI_14"]), 2),
            "a_atr_14_pct": round(float(entry_row["A_ATR_14_Pct"]), 3),
            "a_histvol_20": round(float(entry_row["A_HistVol_20"]), 2),
            "a_vol_ratio_20": round(float(entry_row["A_Volume_Ratio_20"]), 3),
            "b_sma_20_ratio": round(float(entry_row["B_SMA_20_Ratio"]), 4),
            "b_sma_50_ratio": round(float(entry_row["B_SMA_50_Ratio"]), 4),
            "b_sma_200_ratio": round(float(entry_row["B_SMA_200_Ratio"]), 4),
            "b_rsi_14": round(float(entry_row["B_RSI_14"]), 2),
            "b_atr_14_pct": round(float(entry_row["B_ATR_14_Pct"]), 3),
            "b_histvol_20": round(float(entry_row["B_HistVol_20"]), 2),
            "b_vol_ratio_20": round(float(entry_row["B_Volume_Ratio_20"]), 3),
            "rolling_corr_90": round(float(entry_row["Rolling_Corr_90"]), 4),
            "spread_vol_pct_90": round(float(entry_row["Spread_Vol_Pct_90"]), 3),
            "vol_ratio_a_to_b": round(float(entry_row["Vol_Ratio_A_to_B"]), 3),
            "rsi_diff_a_minus_b": round(float(entry_row["RSI_Diff_A_minus_B"]), 2),
            # Forward 7-day trajectory data
            "forward_7d": forward_7d
        }
        return trade_record

    def run_full_extraction_pipeline(self) -> Dict[str, Any]:
        """Runs the complete extraction pipeline over all 156 pairs."""
        logger.info("Initializing Phase 1 Pair Trade Extraction Pipeline...")
        PAIR_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

        if not CURATED_PAIRS_FILE.exists():
            raise FileNotFoundError(f"curated_pairs.json not found at {CURATED_PAIRS_FILE}")

        with open(CURATED_PAIRS_FILE, "r", encoding="utf-8") as f:
            pairs_data = json.load(f)

        pairs = pairs_data.get("pairs", [])
        logger.info(f"Loaded {len(pairs)} curated pairs from {CURATED_PAIRS_FILE}")

        all_trades: List[Dict[str, Any]] = []
        forward_trajectories_map: Dict[str, List[Dict[str, Any]]] = {}

        for p_idx, pair_meta in enumerate(pairs):
            t_a = pair_meta["ticker_a"]
            t_b = pair_meta["ticker_b"]
            pair_df = self.compute_pair_time_series(t_a, t_b)
            if pair_df is None:
                continue

            # Iterate through bars starting after lookback window
            # Step forward skipping active trade holding window to prevent redundant entries
            curr_i = self.lookback_days
            n_len = len(pair_df)
            while curr_i < n_len - 8:
                trade = self.simulate_trade_from_entry(pair_df, curr_i, pair_meta)
                if trade is not None:
                    trade_key = f"{trade['pair_id']}_{trade['entry_date']}"
                    forward_trajectories_map[trade_key] = trade["forward_7d"]
                    all_trades.append(trade)
                    # Advance by holding days to avoid overlapping identical signal on same pair
                    curr_i += max(1, trade["holding_days"])
                else:
                    curr_i += 1

        logger.info(f"Extracted {len(all_trades)} total institutional pair trades across 5 years.")

        # Build DataFrames
        df_all = pd.DataFrame(all_trades)
        if df_all.empty:
            logger.warning("No trades extracted.")
            return {"status": "EMPTY"}

        # Separate forward_7d from flat CSV output for clean ML feature consumption
        df_csv = df_all.drop(columns=["forward_7d"])

        # Train / Test split
        df_train = df_csv[df_csv["split_type"] == "TRAIN"].reset_index(drop=True)
        df_test = df_csv[df_csv["split_type"] == "TEST"].reset_index(drop=True)

        # File paths
        all_csv_path = PAIR_STRATEGY_DIR / "pair_trades_all.csv"
        train_csv_path = PAIR_STRATEGY_DIR / "pair_trades_train.csv"
        test_csv_path = PAIR_STRATEGY_DIR / "pair_trades_test.csv"
        metadata_path = PAIR_STRATEGY_DIR / "pair_trades_metadata.json"
        forward_json_path = PAIR_STRATEGY_DIR / "forward_7d_trajectories.json"

        df_csv.to_csv(all_csv_path, index=False)
        df_train.to_csv(train_csv_path, index=False)
        df_test.to_csv(test_csv_path, index=False)

        with open(forward_json_path, "w", encoding="utf-8") as f:
            json.dump(forward_trajectories_map, f, indent=2)

        # Metrics calculation
        train_win_rate = (len(df_train[df_train["outcome_label"] == 1]) / max(1, len(df_train))) * 100.0
        test_win_rate = (len(df_test[df_test["outcome_label"] == 1]) / max(1, len(df_test))) * 100.0
        avg_holding_train = float(df_train["holding_days"].mean()) if not df_train.empty else 0.0
        avg_holding_test = float(df_test["holding_days"].mean()) if not df_test.empty else 0.0
        avg_return_train = float(df_train["net_return_pct"].mean()) if not df_train.empty else 0.0
        avg_return_test = float(df_test["net_return_pct"].mean()) if not df_test.empty else 0.0

        metadata = {
            "status": "SUCCESS",
            "total_trades": len(df_all),
            "train_trades_count": len(df_train),
            "test_trades_count": len(df_test),
            "split_date_cutoff": self.split_cutoff_date,
            "train_date_range": [str(df_train['entry_date'].min()), str(df_train['entry_date'].max())] if not df_train.empty else [],
            "test_date_range": [str(df_test['entry_date'].min()), str(df_test['entry_date'].max())] if not df_test.empty else [],
            "train_win_rate_pct": round(train_win_rate, 2),
            "test_win_rate_pct": round(test_win_rate, 2),
            "avg_holding_days_train": round(avg_holding_train, 2),
            "avg_holding_days_test": round(avg_holding_test, 2),
            "avg_return_pct_train": round(avg_return_train, 2),
            "avg_return_pct_test": round(avg_return_test, 2),
            "exit_reasons_breakdown": {
                "train": df_train["exit_reason"].value_counts().to_dict() if not df_train.empty else {},
                "test": df_test["exit_reason"].value_counts().to_dict() if not df_test.empty else {}
            },
            "outcome_labels_breakdown": {
                "train": df_train["outcome_text"].value_counts().to_dict() if not df_train.empty else {},
                "test": df_test["outcome_text"].value_counts().to_dict() if not df_test.empty else {}
            }
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Extraction pipeline complete. Artifacts written:")
        logger.info(f" - All Trades: {all_csv_path} ({len(df_csv)} records)")
        logger.info(f" - Train 4Y: {train_csv_path} ({len(df_train)} records | Win Rate: {train_win_rate:.1f}% | Avg Holding: {avg_holding_train:.1f}d)")
        logger.info(f" - Test 1Y (Holdout): {test_csv_path} ({len(df_test)} records | Win Rate: {test_win_rate:.1f}% | Avg Holding: {avg_holding_test:.1f}d)")
        logger.info(f" - 7D Forward Paths: {forward_json_path} ({len(forward_trajectories_map)} cached)")
        logger.info(f" - Metadata: {metadata_path}")

        return metadata


if __name__ == "__main__":
    engine = PairTradeEngine()
    meta = engine.run_full_extraction_pipeline()
    print("\n--- Phase 1 Extraction Summary ---")
    print(json.dumps(meta, indent=2))
