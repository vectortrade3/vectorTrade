"""
VectorTrade - Setup Quality Classification Model Trainer
Trains 3-Tier Setup Quality Classification Models for Active Strategies:
1. 50 MA Setup Strategy (ma50_setup)
2. Global Regime Strategy (global_regime)

Strict Constraints:
- Trained and tested ONLY on setups (Base_Signal / triggered setups), NOT continuous data.
- Strict 4-year train / 1-year test chronological split (Zero lookahead / data leakage).
- 3 Tiers:
    * Class 2 ("Very Good"): Hit >= 1:2 R:R Target (Outcome == 'SUCCESS' / R >= 2.0)
    * Class 1 ("Moderate"): Hit 1:1 R:R Target (1.0 <= R < 2.0)
    * Class 0 ("Poor / Bad"): Hit Stop Loss (Outcome == 'LOSS' / R < 1.0)
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

logger = logging.getLogger("VectorTrade.QualityTrainer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Strict leakage columns to shield from feature matrices
LEAKAGE_COLS = [
    'base_signal', 'signal_strength', 'entry_price', 'stop_loss', 'risk', 'target_1_2',
    'target_hit', 'sl_hit', 'timeout', 'outcome', 'days_to_outcome',
    'return_7d_after_signal', 'max_gain_7d', 'max_loss_7d', 'mfe', 'mae', 'Trade_Triggered',
    'max_r_achieved', 'date', 'symbol', 'sector', 'Stock Symbol', 'Sector', 'Trading_Day_Of_Week',
    'trade_id', 'setup_date', 'setup_num', 'setup_name',
    'sl_price', 'target_price', 'risk_reward', 'risk_reward_num',
    'loss_pct', 'profit_pct', 'triggered', 'days_held', 'exit_price', 'exit_date',
    'realized_pnl_pct', 'realized_r', 'mfe_pct', 'mae_pct',
    'chart_candles_json', 'split', 'is_test_1y', 'candle_green', 'bb_near_low',
    'setup_open', 'setup_high', 'setup_low', 'setup_close', 'setup_volume',
    'pred_class', 'prob_class0', 'prob_class1', 'prob_class2'
]


def _filter_features(df: pd.DataFrame, max_null_ratio: float = 0.4) -> List[str]:
    """Extracts valid numeric features excluding all leakage and future lookahead columns."""
    valid_cols = []
    for c in df.columns:
        if c in LEAKAGE_COLS or c.startswith('fwd_') or c.startswith('Setup_'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].isnull().mean() <= max_null_ratio:
                valid_cols.append(c)
    return sorted(valid_cols)


def train_ma50_quality_classifier() -> Dict[str, Any]:
    """
    Trains the 3-Tier Quality Classification Model for 50 MA Setup Strategy:
    - 4-Year Train Set: 5_year_final_data_50ma_train.csv (7,091 setups, 2021-11 to 2025-10)
    - 1-Year Test Set:  5_year_final_data_50ma_test_1y.csv (1,576 setups, 2025-10 to 2026-10)
    """
    logger.info("Training 50 MA Setup Quality Classifier...")
    train_path = PROJECT_ROOT / "5_year_final_data_50ma_train.csv"
    test_path = PROJECT_ROOT / "5_year_final_data_50ma_test_1y.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing 50MA datasets at {train_path} or {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 3-Tier Target Labeling
    def _get_target(df: pd.DataFrame) -> np.ndarray:
        c2 = (df['realized_r'] >= 2.0) | (df['outcome'] == 'SUCCESS')
        c1 = (~c2) & ((df['realized_r'] >= 1.0) | (df['mfe_pct'] >= df['loss_pct']))
        y = np.zeros(len(df), dtype=int)
        y[c1] = 1
        y[c2] = 2
        return y

    y_train = _get_target(train_df)
    y_test = _get_target(test_df)

    feats_train = _filter_features(train_df)
    feats = [f for f in feats_train if f in test_df.columns]

    X_train_raw = train_df[feats].replace([np.inf, -np.inf], np.nan)
    X_test_raw = test_df[feats].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_test_s = scaler.transform(imputer.transform(X_test_raw))

    # XGBoost 3-class classifier
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss'
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

    # Class-specific metrics
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Thresholded performance for Class 2 ("Very Good")
    prob_c2 = y_prob[:, 2]
    threshold_analysis = []
    base_win_rate = float((y_test == 2).mean())

    for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
        mask = (prob_c2 >= th)
        n_trades = int(mask.sum())
        if n_trades > 0:
            win_pct = float((y_test[mask] == 2).mean() * 100.0)
            loss_pct = float((y_test[mask] == 0).mean() * 100.0)
            avg_r = float(test_df.loc[mask, 'realized_r'].mean())
        else:
            win_pct = 0.0
            loss_pct = 0.0
            avg_r = 0.0
        threshold_analysis.append({
            "threshold": th,
            "selected_trades_count": n_trades,
            "win_rate_pct": round(win_pct, 2),
            "loss_rate_pct": round(loss_pct, 2),
            "avg_realized_r": round(avg_r, 2)
        })

    # Save artifacts
    model_file = MODELS_DIR / "ma50_quality_classifier.pkl"
    scaler_file = MODELS_DIR / "ma50_quality_scaler.pkl"
    imputer_file = MODELS_DIR / "ma50_quality_imputer.pkl"
    feats_file = MODELS_DIR / "ma50_quality_features.txt"

    joblib.dump(clf, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(imputer, imputer_file)
    with open(feats_file, "w", encoding="utf-8") as f:
        for feat in feats:
            f.write(f"{feat}\n")

    logger.info(f"Saved 50MA Quality Classifier artifacts to {MODELS_DIR}")

    return {
        "strategy": "50 MA Setup Strategy",
        "model_file": str(model_file.name),
        "train_samples_count": len(train_df),
        "test_samples_count": len(test_df),
        "features_count": len(feats),
        "baseline_target_win_rate": round(base_win_rate * 100.0, 2),
        "test_accuracy": round(acc * 100.0, 2),
        "macro_precision": round(prec_macro * 100.0, 2),
        "macro_recall": round(rec_macro * 100.0, 2),
        "macro_f1": round(f1_macro * 100.0, 2),
        "per_class_metrics": {
            "Poor (Class 0)": {
                "precision": round(prec_per_class[0] * 100.0, 2),
                "recall": round(rec_per_class[0] * 100.0, 2),
                "f1": round(f1_per_class[0] * 100.0, 2)
            },
            "Moderate (Class 1)": {
                "precision": round(prec_per_class[1] * 100.0, 2),
                "recall": round(rec_per_class[1] * 100.0, 2),
                "f1": round(f1_per_class[1] * 100.0, 2)
            },
            "Very Good (Class 2)": {
                "precision": round(prec_per_class[2] * 100.0, 2),
                "recall": round(rec_per_class[2] * 100.0, 2),
                "f1": round(f1_per_class[2] * 100.0, 2)
            }
        },
        "confusion_matrix": cm,
        "threshold_analysis": threshold_analysis
    }


def train_global_regime_quality_classifier() -> Dict[str, Any]:
    """
    Trains the 3-Tier Quality Classification Model for Global Regime Strategy:
    - Master Setups Dataset: 5_year_final_data.csv (8,720 setups with forward 7d targets)
    - 4-Year Train Set: Setups before 2025-10-06 (7,125 setups)
    - 1-Year Test Set:  Setups from 2025-10-06 to 2026-10-05 (1,595 setups)
    """
    logger.info("Training Global Regime Setup Quality Classifier...")
    data_path = PROJECT_ROOT / "5_year_final_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing master dataset at {data_path}")

    df = pd.read_csv(data_path)
    valid = df.dropna(subset=['max_gain_7d', 'max_loss_7d']).copy()
    valid['date'] = pd.to_datetime(valid['date'])
    valid = valid.sort_values('date').reset_index(drop=True)

    train_mask = valid['date'] < '2025-10-06'
    test_mask = valid['date'] >= '2025-10-06'

    train_df = valid[train_mask].copy()
    test_df = valid[test_mask].copy()

    # 3-Tier Target Labeling for Global Regime:
    # Risk = max(max_loss_7d, close * 0.015)
    # R_mult = max_gain_7d / Risk
    # Class 2: R_mult >= 2.0 (Target 1:2 hit)
    # Class 1: 1.0 <= R_mult < 2.0 (Target 1:1 hit)
    # Class 0: R_mult < 1.0 (Stop Loss hit / drop exceeded gain)
    def _get_target(d: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        risk = np.maximum(d['max_loss_7d'], d['close'] * 0.015)
        r_mult = d['max_gain_7d'] / risk
        c2 = r_mult >= 2.0
        c1 = (~c2) & (r_mult >= 1.0)
        y = np.zeros(len(d), dtype=int)
        y[c1] = 1
        y[c2] = 2
        return y, r_mult

    y_train, r_train = _get_target(train_df)
    y_test, r_test = _get_target(test_df)
    test_df['realized_r'] = r_test

    feats_train = _filter_features(train_df)
    feats = [f for f in feats_train if f in test_df.columns]

    X_train_raw = train_df[feats].replace([np.inf, -np.inf], np.nan)
    X_test_raw = test_df[feats].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_test_s = scaler.transform(imputer.transform(X_test_raw))

    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss'
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_test, y_pred).tolist()

    prob_c2 = y_prob[:, 2]
    threshold_analysis = []
    base_win_rate = float((y_test == 2).mean())

    for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
        mask = (prob_c2 >= th)
        n_trades = int(mask.sum())
        if n_trades > 0:
            win_pct = float((y_test[mask] == 2).mean() * 100.0)
            loss_pct = float((y_test[mask] == 0).mean() * 100.0)
            avg_r = float(test_df.loc[mask, 'realized_r'].mean())
        else:
            win_pct = 0.0
            loss_pct = 0.0
            avg_r = 0.0
        threshold_analysis.append({
            "threshold": th,
            "selected_trades_count": n_trades,
            "win_rate_pct": round(win_pct, 2),
            "loss_rate_pct": round(loss_pct, 2),
            "avg_realized_r": round(avg_r, 2)
        })

    # Save artifacts
    model_file = MODELS_DIR / "global_regime_quality_classifier.pkl"
    scaler_file = MODELS_DIR / "global_regime_quality_scaler.pkl"
    imputer_file = MODELS_DIR / "global_regime_quality_imputer.pkl"
    feats_file = MODELS_DIR / "global_regime_quality_features.txt"

    joblib.dump(clf, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(imputer, imputer_file)
    with open(feats_file, "w", encoding="utf-8") as f:
        for feat in feats:
            f.write(f"{feat}\n")

    logger.info(f"Saved Global Regime Quality Classifier artifacts to {MODELS_DIR}")

    return {
        "strategy": "Global Regime Strategy",
        "model_file": str(model_file.name),
        "train_samples_count": len(train_df),
        "test_samples_count": len(test_df),
        "features_count": len(feats),
        "baseline_target_win_rate": round(base_win_rate * 100.0, 2),
        "test_accuracy": round(acc * 100.0, 2),
        "macro_precision": round(prec_macro * 100.0, 2),
        "macro_recall": round(rec_macro * 100.0, 2),
        "macro_f1": round(f1_macro * 100.0, 2),
        "per_class_metrics": {
            "Poor (Class 0)": {
                "precision": round(prec_per_class[0] * 100.0, 2),
                "recall": round(rec_per_class[0] * 100.0, 2),
                "f1": round(f1_per_class[0] * 100.0, 2)
            },
            "Moderate (Class 1)": {
                "precision": round(prec_per_class[1] * 100.0, 2),
                "recall": round(rec_per_class[1] * 100.0, 2),
                "f1": round(f1_per_class[1] * 100.0, 2)
            },
            "Very Good (Class 2)": {
                "precision": round(prec_per_class[2] * 100.0, 2),
                "recall": round(rec_per_class[2] * 100.0, 2),
                "f1": round(f1_per_class[2] * 100.0, 2)
            }
        },
        "confusion_matrix": cm,
        "threshold_analysis": threshold_analysis
    }


def train_all_quality_models() -> Dict[str, Any]:
    """Trains both 50MA and Global Regime quality classification models and returns evaluation report."""
    ma50_report = train_ma50_quality_classifier()
    global_report = train_global_regime_quality_classifier()

    full_report = {
        "ma50_setup": ma50_report,
        "global_regime": global_report,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = PROJECT_ROOT / "quality_models_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Evaluation report written to {report_path}")
    return full_report


if __name__ == "__main__":
    results = train_all_quality_models()
    print("\n" + "="*80)
    print("QUALITY CLASSIFICATION MODELS EVALUATION RESULTS (1-YEAR OUT-OF-SAMPLE TEST)")
    print("="*80)
    print(json.dumps(results, indent=2))
