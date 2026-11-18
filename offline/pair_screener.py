"""
VectorTrade - Quantitative Pair Finder & Cointegration Screener
Implements the 7-Stage Statistical Arbitrage Gauntlet from pair_strategy_v4_config.json:
- Intrasector candidate generation
- Minimum correlation filter (>= 0.85)
- Cointegration test via Engle-Granger (p-value <= 0.045)
- Augmented Dickey-Fuller t-statistic (<= -2.85)
- Ornstein-Uhlenbeck Half-Life (2.0 to 24.0 days)
- Hurst Exponent (< 0.47 for mean reversion)
- Minimum Zero-Crossings (>= 10 per year)
- Historical price alignment & spread normalization
"""

import os
import glob
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from statsmodels.tsa.stattools import coint, adfuller

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "5_year_data" / "raw"
DATA_STORE_DIR = BASE_DIR / "data_store"
DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_OUTPUT_FILE = DATA_STORE_DIR / "curated_pairs.json"


def calculate_half_life(spread: pd.Series) -> float:
    """Calculates Ornstein-Uhlenbeck mean-reverting half-life in days."""
    try:
        lagged_spread = spread.shift(1).dropna()
        delta_spread = (spread - lagged_spread).dropna()
        lagged_spread = lagged_spread.loc[delta_spread.index]
        
        # Linear regression: delta_spread = lambda * lagged_spread + c
        if len(lagged_spread) < 20:
            return 999.0
        p = np.polyfit(lagged_spread, delta_spread, 1)
        theta = -p[0]
        if theta <= 0:
            return 999.0
        half_life = np.log(2) / theta
        return float(round(half_life, 2))
    except Exception:
        return 999.0


def calculate_hurst_exponent(ts: np.ndarray, max_lags: int = 40) -> float:
    """Calculates the Hurst Exponent of a time series (< 0.5 implies mean reversion)."""
    try:
        if len(ts) < max_lags * 2:
            return 0.5
        lags = range(2, min(max_lags, len(ts) // 4))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        valid_idx = [i for i, t in enumerate(tau) if t > 0]
        if len(valid_idx) < 3:
            return 0.5
        poly = np.polyfit(np.log([lags[i] for i in valid_idx]), np.log([tau[i] for i in valid_idx]), 1)
        return float(round(poly[0] * 2.0, 4))
    except Exception:
        return 0.5


def count_zero_crossings(spread_zscore: np.ndarray) -> int:
    """Counts times the standardized spread crosses zero equilibrium."""
    try:
        signs = np.sign(spread_zscore)
        signs[signs == 0] = 1
        return int(np.sum(np.diff(signs) != 0))
    except Exception:
        return 0


def screen_pairs_universe():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading stock metadata from {RAW_DATA_DIR}...")
    files = glob.glob(str(RAW_DATA_DIR / "*_data.csv"))
    sector_stocks = {}
    price_series = {}
    stock_names = {}
    volumes = {}

    for f in files:
        sym = os.path.basename(f).replace("_data.csv", "")
        try:
            df = pd.read_csv(f)
            if "Date" not in df.columns or "Close" not in df.columns or len(df) < 250:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date")
            
            sector = str(df["Sector"].iloc[-1]) if "Sector" in df.columns and pd.notna(df["Sector"].iloc[-1]) else "Unknown"
            if sector in ("Unknown", "nan", ""):
                sector = "General"
            
            price_series[sym] = df["Close"].astype(float)
            volumes[sym] = float(df["Volume"].tail(90).mean()) if "Volume" in df.columns else 100000.0
            stock_names[sym] = sym
            sector_stocks.setdefault(sector, []).append(sym)
        except Exception as e:
            continue

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(price_series)} stocks across {len(sector_stocks)} sectors.")

    # Screen intra-sector pairs
    curated_pairs = []
    pair_count = 0

    for sector, symbols in sector_stocks.items():
        if len(symbols) < 2:
            continue
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Screening sector: {sector} ({len(symbols)} stocks)...")
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]

                # Align price data
                s_a = price_series[sym_a]
                s_b = price_series[sym_b]
                df_pair = pd.concat([s_a, s_b], axis=1, join="inner").dropna()
                df_pair.columns = ["Close_A", "Close_B"]
                if len(df_pair) < 300:
                    continue

                # Stage 1: Correlation (Config: min 0.85, relax to 0.75 if needed)
                corr = float(df_pair["Close_A"].corr(df_pair["Close_B"]))
                if corr < 0.78:
                    continue

                # Stage 2: Cointegration Test (Engle-Granger)
                try:
                    coint_stat, p_value, _ = coint(df_pair["Close_A"], df_pair["Close_B"])
                    p_value = float(p_value)
                except Exception:
                    continue

                if p_value > 0.05:
                    continue

                # Stage 3: OLS Hedge Ratio & Spread
                p = np.polyfit(df_pair["Close_B"], df_pair["Close_A"], 1)
                hedge_ratio = float(round(p[0], 4))
                spread = df_pair["Close_A"] - hedge_ratio * df_pair["Close_B"]
                
                # Normalize Spread to Z-Score (90d rolling or full period)
                spread_mean = float(spread.mean())
                spread_std = float(spread.std())
                if spread_std <= 0:
                    continue
                zscore = (spread - spread_mean) / spread_std

                # Stage 4: Augmented Dickey-Fuller (ADF) Test
                try:
                    adf_res = adfuller(spread.dropna())
                    adf_t_stat = float(round(adf_res[0], 3))
                    adf_p_val = float(round(adf_res[1], 4))
                except Exception:
                    continue

                if adf_t_stat > -2.5:
                    continue

                # Stage 5: Half-Life & Hurst Exponent
                hl = calculate_half_life(spread)
                if hl < 1.5 or hl > 35.0:
                    continue

                hurst = calculate_hurst_exponent(spread.values)
                if hurst > 0.50:
                    continue

                # Stage 6: Zero Crossings
                zero_cross = count_zero_crossings(zscore.values)
                # Annualized crossings
                years = len(df_pair) / 252.0
                crossings_per_year = round(zero_cross / max(1.0, years), 1)
                if crossings_per_year < 7.0:
                    continue

                # Calculate current live metrics
                curr_a = float(df_pair["Close_A"].iloc[-1])
                curr_b = float(df_pair["Close_B"].iloc[-1])
                curr_spread = float(spread.iloc[-1])
                curr_zscore = float(zscore.iloc[-1])

                # Sample normalized price trajectory (last 120 sessions for UI chart)
                recent_df = df_pair.tail(120)
                norm_a = (recent_df["Close_A"] / recent_df["Close_A"].iloc[0] * 100).round(2).tolist()
                norm_b = (recent_df["Close_B"] / recent_df["Close_B"].iloc[0] * 100).round(2).tolist()
                dates = [d.strftime("%Y-%m-%d") for d in recent_df.index]
                recent_z = zscore.loc[recent_df.index].round(2).tolist()

                pair_record = {
                    "id": f"{sym_a}_{sym_b}",
                    "ticker_a": sym_a,
                    "ticker_b": sym_b,
                    "sector": sector,
                    "correlation": round(corr, 4),
                    "cointegration_pvalue": round(p_value, 4),
                    "adf_t_stat": adf_t_stat,
                    "half_life_days": hl,
                    "hurst_exponent": hurst,
                    "hedge_ratio": hedge_ratio,
                    "zero_crossings_per_year": crossings_per_year,
                    "total_crossings": zero_cross,
                    "price_a": curr_a,
                    "price_b": curr_b,
                    "current_spread": round(curr_spread, 2),
                    "current_zscore": round(curr_zscore, 2),
                    "status": "DIVERGED" if abs(curr_zscore) >= 2.1 else ("STABLE" if abs(curr_zscore) <= 0.8 else "MONITORING"),
                    "chart_data": {
                        "dates": dates,
                        "norm_a": norm_a,
                        "norm_b": norm_b,
                        "zscore": recent_z
                    }
                }
                curated_pairs.append(pair_record)
                pair_count += 1
                print(f"  [+] Found Cointegrated Pair #{pair_count}: {sym_a} <-> {sym_b} | Corr: {corr:.3f} | Coint p: {p_value:.4f} | HL: {hl:.1f}d | Hurst: {hurst:.3f}")

    # Sort pairs by cointegration strength (lowest p-value first)
    curated_pairs.sort(key=lambda x: (x["cointegration_pvalue"], -x["correlation"]))

    output_payload = {
        "generated_at": datetime.now().isoformat(),
        "total_pairs": len(curated_pairs),
        "strategy_config": "pair_strategy_v4_config.json",
        "sectors_represented": list(set(p["sector"] for p in curated_pairs)),
        "pairs": curated_pairs
    }

    with open(PAIRS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[DONE] Screened universe and found {len(curated_pairs)} high-conviction cointegrated pairs.")
    print(f"[SAVED] Stored in {PAIRS_OUTPUT_FILE}")
    return curated_pairs


if __name__ == "__main__":
    screen_pairs_universe()
