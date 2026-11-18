"""
VectorTrade - Simple Intra-Sector Statistical Pairs Formation
Builds natural pairs using purely fundamental and statistical parameters without any external strategy config:
1. Intra-Sector: Stocks must belong to the exact same industry/sector.
2. 5-Year Data Length: Minimum 300 overlapping daily sessions.
3. High Historical Price Correlation (r >= 0.85): Direct measure of co-movement.
4. Recent 1-Year Confirmation (r_1y >= 0.65): Ensures pair relationship is actively co-moving.
5. Standard Normalized Price series & Spread Z-Score for interactive visual inspection.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "5_year_data" / "raw"
DATA_STORE_DIR = BASE_DIR / "data_store"
DATA_STORE_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_OUTPUT_FILE = DATA_STORE_DIR / "curated_pairs.json"


def form_pairs_without_config(min_5y_corr: float = 0.85, min_1y_corr: float = 0.65):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading 5-year verified price series from {RAW_DATA_DIR}...")
    files = glob.glob(str(RAW_DATA_DIR / "*_data.csv"))
    
    sector_stocks = {}
    price_series = {}

    for f in files:
        sym = os.path.basename(f).replace("_data.csv", "")
        try:
            df = pd.read_csv(f)
            if "Date" not in df.columns or "Close" not in df.columns or len(df) < 300:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
            
            sector = str(df["Sector"].iloc[-1]) if "Sector" in df.columns and pd.notna(df["Sector"].iloc[-1]) else "General"
            if sector in ("Unknown", "nan", ""):
                sector = "General"
            
            price_series[sym] = df["Close"].astype(float)
            sector_stocks.setdefault(sector, []).append(sym)
        except Exception:
            continue

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(price_series)} stocks across {len(sector_stocks)} sectors.")

    all_pairs = []

    for sector, symbols in sector_stocks.items():
        if len(symbols) < 2:
            continue
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Forming pairs for sector: {sector} ({len(symbols)} stocks)...")
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]

                s_a = price_series[sym_a]
                s_b = price_series[sym_b]
                df_pair = pd.concat([s_a, s_b], axis=1, join="inner").dropna()
                df_pair.columns = ["Close_A", "Close_B"]
                if len(df_pair) < 300:
                    continue

                # 1. Full 5-Year Pearson Correlation
                corr_5y = float(df_pair["Close_A"].corr(df_pair["Close_B"]))
                if corr_5y < min_5y_corr:
                    continue

                # 2. Recent 1-Year (last 252 bars) Correlation confirmation
                recent_1y = df_pair.tail(252)
                corr_1y = float(recent_1y["Close_A"].corr(recent_1y["Close_B"])) if len(recent_1y) >= 60 else corr_5y
                if corr_1y < min_1y_corr:
                    continue

                # 3. Simple OLS Hedge Ratio (beta): Stock A = beta * Stock B
                p = np.polyfit(df_pair["Close_B"], df_pair["Close_A"], 1)
                hedge_ratio = float(round(p[0], 4))
                spread = df_pair["Close_A"] - hedge_ratio * df_pair["Close_B"]

                # 4. Standard Spread Statistics (Mean, Std, Current Z-Score)
                spread_mean = float(spread.mean())
                spread_std = float(spread.std())
                if spread_std <= 0:
                    continue
                zscore = (spread - spread_mean) / spread_std
                curr_zscore = float(zscore.iloc[-1])

                curr_a = float(df_pair["Close_A"].iloc[-1])
                curr_b = float(df_pair["Close_B"].iloc[-1])
                curr_spread = float(spread.iloc[-1])

                # Sample normalized price trajectory (last 120 trading sessions for visual chart)
                recent_120 = df_pair.tail(120)
                norm_a = (recent_120["Close_A"] / recent_120["Close_A"].iloc[0] * 100).round(2).tolist()
                norm_b = (recent_120["Close_B"] / recent_120["Close_B"].iloc[0] * 100).round(2).tolist()
                dates = [d.strftime("%Y-%m-%d") for d in recent_120.index]
                recent_z = zscore.loc[recent_120.index].round(2).tolist()

                # Status based purely on Z-score deviation
                if abs(curr_zscore) >= 2.0:
                    status = "DIVERGED"
                elif abs(curr_zscore) <= 0.8:
                    status = "IN_SYNC"
                else:
                    status = "MONITORING"

                pair_record = {
                    "id": f"{sym_a}_{sym_b}",
                    "ticker_a": sym_a,
                    "ticker_b": sym_b,
                    "sector": sector,
                    "correlation_5y": round(corr_5y, 4),
                    "correlation_1y": round(corr_1y, 4),
                    "correlation": round(corr_5y, 4),  # for template compatibility
                    "hedge_ratio": hedge_ratio,
                    "price_a": curr_a,
                    "price_b": curr_b,
                    "current_spread": round(curr_spread, 2),
                    "current_zscore": round(curr_zscore, 2),
                    "status": status,
                    "data_points": len(df_pair),
                    "chart_data": {
                        "dates": dates,
                        "norm_a": norm_a,
                        "norm_b": norm_b,
                        "zscore": recent_z
                    }
                }
                all_pairs.append(pair_record)

    # Sort pairs by 5Y correlation descending
    all_pairs.sort(key=lambda x: -x["correlation_5y"])

    output_payload = {
        "generated_at": datetime.now().isoformat(),
        "method": "Simple Intra-Sector Statistical Co-Movement",
        "parameters": {
            "intra_sector_only": True,
            "min_5y_correlation": min_5y_corr,
            "min_1y_correlation": min_1y_corr,
            "min_history_bars": 300
        },
        "total_pairs": len(all_pairs),
        "sectors_represented": sorted(list(set(p["sector"] for p in all_pairs))),
        "pairs": all_pairs
    }

    # Save to data_store/curated_pairs.json
    with open(PAIRS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[DONE] Successfully formed {len(all_pairs)} intra-sector pairs based on simple statistical correlation!")
    print(f"[SAVED] Stored in {PAIRS_OUTPUT_FILE}")
    return all_pairs


if __name__ == "__main__":
    form_pairs_without_config()
