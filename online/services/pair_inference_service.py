"""
Institutional Cointegrated Pairs Trading Strategy v4 - Online / Daily Inference Service
Module: online/services/pair_inference_service.py

Performs:
1. Daily end-of-day scanning of all 156 curated pairs in data_store/curated_pairs.json.
2. Extracts current spread Z-score, rolling OLS beta, and divergence momentum slope (Delta Z).
3. Constructs the 23-feature vector for pairs meeting divergence criteria (|Z| in [2.10, 3.00], |Delta Z| <= 0.05).
4. Runs calibrated inference through models/pair_trading_quality_classifier.pkl.
5. Qualifies high-conviction pair trade setups (predicted P(Win) >= optimal threshold).
6. Persists qualified setups to data_store/pair_strategy/daily_pair_signals.json and optionally syncs to MongoDB.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PairInferenceService")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_STORE_DIR = BASE_DIR / "data_store"
RAW_DATA_DIR = BASE_DIR / "5_year_data" / "raw"
MODELS_DIR = BASE_DIR / "models"
PAIR_STRATEGY_DIR = DATA_STORE_DIR / "pair_strategy"
CURATED_PAIRS_FILE = DATA_STORE_DIR / "curated_pairs.json"


class PairInferenceService:
    """Service for real-time daily inference and screening of pairs trading signals."""

    _model = None
    _scaler = None
    _imputer = None
    _features: List[str] = []
    _optimal_threshold: float = 0.53

    @classmethod
    def load_artifacts(cls) -> bool:
        """Loads model, preprocessors, and features list."""
        if cls._model is not None:
            return True

        model_path = MODELS_DIR / "pair_trading_quality_classifier.pkl"
        scaler_path = MODELS_DIR / "pair_trading_scaler.pkl"
        imputer_path = MODELS_DIR / "pair_trading_imputer.pkl"
        features_path = MODELS_DIR / "pair_trading_features.txt"
        metrics_path = MODELS_DIR / "pair_trading_model_metrics.json"

        if not (model_path.exists() and scaler_path.exists() and imputer_path.exists() and features_path.exists()):
            logger.warning("Pair trading model artifacts not yet built in models/")
            return False

        try:
            with open(model_path, "rb") as f:
                cls._model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                cls._scaler = pickle.load(f)
            with open(imputer_path, "rb") as f:
                cls._imputer = pickle.load(f)
            with open(features_path, "r", encoding="utf-8") as f:
                cls._features = [line.strip() for line in f if line.strip()]

            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                    cls._optimal_threshold = float(metrics.get("optimal_threshold", 0.53))

            logger.info(f"Loaded PairTradingQualityClassifier with {len(cls._features)} features (Optimal Threshold: {cls._optimal_threshold})")
            return True
        except Exception as e:
            logger.error(f"Error loading pair trading artifacts: {e}")
            return False

    @classmethod
    def evaluate_daily_pairs(cls, min_prob: Optional[float] = None) -> Dict[str, Any]:
        """
        Scans all curated pairs against current market closing data,
        computes dual-stock indicators and spread metrics,
        and generates ML-qualified signals.
        """
        if not cls.load_artifacts():
            return {"status": "ERROR", "message": "Model artifacts unavailable", "signals": []}

        if not CURATED_PAIRS_FILE.exists():
            return {"status": "ERROR", "message": "curated_pairs.json missing", "signals": []}

        threshold = min_prob if min_prob is not None else cls._optimal_threshold

        with open(CURATED_PAIRS_FILE, "r", encoding="utf-8") as f:
            pairs_data = json.load(f)
        pairs = pairs_data.get("pairs", [])

        from offline.pair_trade_engine import PairTradeEngine, compute_half_life, compute_adf_stat
        engine = PairTradeEngine()

        candidate_setups: List[Dict[str, Any]] = []

        for p_meta in pairs:
            t_a = p_meta["ticker_a"]
            t_b = p_meta["ticker_b"]
            df_pair = engine.compute_pair_time_series(t_a, t_b)
            if df_pair is None or len(df_pair) < 95:
                continue

            last_row = df_pair.iloc[-1]
            prev_row = df_pair.iloc[-2]

            z_curr = float(last_row["Z_Score"])
            delta_z = float(last_row["Delta_Z"])

            # Check divergence hurdle: |Z| in [2.10, 3.00] and velocity slope <= 0.05
            is_long = (-3.00 <= z_curr <= -2.10)
            is_short = (2.10 <= z_curr <= 3.00)

            if not (is_long or is_short):
                continue
            if abs(delta_z) > 0.05:
                continue

            trade_dir = "LONG_SPREAD" if is_long else "SHORT_SPREAD"

            # Compute half-life & ADF
            spread_window = df_pair["Spread"].iloc[-90:]
            hl = compute_half_life(spread_window)
            adf = compute_adf_stat(spread_window)

            # Build feature dict
            feat_dict = {
                "z_score_entry": z_curr,
                "delta_z_entry": delta_z,
                "hedge_ratio_beta": float(last_row["Hedge_Ratio_Beta"]),
                "half_life_days": hl,
                "adf_stat": adf,
                "a_sma_20_ratio": float(last_row["A_SMA_20_Ratio"]),
                "a_sma_50_ratio": float(last_row["A_SMA_50_Ratio"]),
                "a_sma_200_ratio": float(last_row["A_SMA_200_Ratio"]),
                "a_rsi_14": float(last_row["A_RSI_14"]),
                "a_atr_14_pct": float(last_row["A_ATR_14_Pct"]),
                "a_histvol_20": float(last_row["A_HistVol_20"]),
                "a_vol_ratio_20": float(last_row["A_Volume_Ratio_20"]),
                "b_sma_20_ratio": float(last_row["B_SMA_20_Ratio"]),
                "b_sma_50_ratio": float(last_row["B_SMA_50_Ratio"]),
                "b_sma_200_ratio": float(last_row["B_SMA_200_Ratio"]),
                "b_rsi_14": float(last_row["B_RSI_14"]),
                "b_atr_14_pct": float(last_row["B_ATR_14_Pct"]),
                "b_histvol_20": float(last_row["B_HistVol_20"]),
                "b_vol_ratio_20": float(last_row["B_Volume_Ratio_20"]),
                "rolling_corr_90": float(last_row["Rolling_Corr_90"]),
                "spread_vol_pct_90": float(last_row["Spread_Vol_Pct_90"]),
                "vol_ratio_a_to_b": float(last_row["Vol_Ratio_A_to_B"]),
                "rsi_diff_a_minus_b": float(last_row["RSI_Diff_A_minus_B"])
            }

            candidate_setups.append({
                "pair_id": p_meta.get("id", f"{t_a}_{t_b}"),
                "ticker_a": t_a,
                "ticker_b": t_b,
                "sector": p_meta.get("sector", "Unknown"),
                "trade_direction": trade_dir,
                "date": str(last_row["Date"]),
                "price_a": float(last_row["Close_A"]),
                "price_b": float(last_row["Close_B"]),
                "hedge_ratio_beta": round(float(last_row["Hedge_Ratio_Beta"]), 4),
                "z_score": round(z_curr, 3),
                "delta_z": round(delta_z, 4),
                "half_life_days": round(hl, 2),
                "features": feat_dict
            })

        qualified_signals: List[Dict[str, Any]] = []

        if candidate_setups:
            # Batch ML probability inference
            X_raw = np.array([[c["features"][f] for f in cls._features] for c in candidate_setups], dtype=np.float32)
            X_scaled = cls._scaler.transform(cls._imputer.transform(X_raw))
            probs = cls._model.predict_proba(X_scaled)[:, 1]

            for cand, prob in zip(candidate_setups, probs):
                cand["ml_win_probability"] = round(float(prob), 4)
                cand["ml_qualified"] = bool(prob >= threshold)
                cand["threshold_used"] = threshold
                if cand["ml_qualified"]:
                    qualified_signals.append(cand)

        # Persist daily signals cache
        PAIR_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
        signals_out = {
            "status": "SUCCESS",
            "evaluated_pairs_count": len(pairs),
            "candidates_found_count": len(candidate_setups),
            "qualified_signals_count": len(qualified_signals),
            "decision_threshold": threshold,
            "qualified_signals": qualified_signals,
            "all_candidates": candidate_setups
        }

        signals_file = PAIR_STRATEGY_DIR / "daily_pair_signals.json"
        with open(signals_file, "w", encoding="utf-8") as f:
            json.dump(signals_out, f, indent=2)

        logger.info(f"Daily Pair Scan: {len(pairs)} pairs -> {len(candidate_setups)} hurdle candidates -> {len(qualified_signals)} ML-qualified signals.")
        return signals_out


pair_inference_service = PairInferenceService()


if __name__ == "__main__":
    res = PairInferenceService.evaluate_daily_pairs()
    print(json.dumps({k: v for k, v in res.items() if k != "all_candidates"}, indent=2))
