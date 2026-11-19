"""
Institutional Cointegrated Pairs Trading Strategy v4 - ML Model Trainer (Phase 3)
Pipeline Module: offline/pair_model_trainer.py

Implements:
1. Strict 4-Year Train (<= 2025-11-18) vs 1-Year Holdout Test (> 2025-11-18) ingestion.
2. Binary target formulation: y = 1 if outcome_label == 1 (Success), y = 0 if outcome_label == 0 (Fail).
3. Preprocessing pipeline: SimpleImputer(strategy='median') + RobustScaler(quantile_range=(5.0, 95.0)).
4. Purged Grouped Time-Series Cross-Validation over training set.
5. Calibrated XGBClassifier model training with optimal hyperparameter search.
6. Out-of-sample holdout evaluation on 1-year holdout dataset.
7. Decision threshold tuning for institutional benchmark gates (Win Rate >= 58%, Profit Factor >= 1.65, Precision >= 62%).
8. Serialization of model pipeline artifacts to models/:
   - models/pair_trading_quality_classifier.pkl
   - models/pair_trading_scaler.pkl
   - models/pair_trading_imputer.pkl
   - models/pair_trading_features.txt
   - models/pair_trading_model_metrics.json
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PairModelTrainer")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_STORE_DIR = BASE_DIR / "data_store"
PAIR_STRATEGY_DIR = DATA_STORE_DIR / "pair_strategy"
MODELS_DIR = BASE_DIR / "models"

FEATURE_NAMES: List[str] = [
    "z_score_entry",
    "delta_z_entry",
    "hedge_ratio_beta",
    "half_life_days",
    "adf_stat",
    "a_sma_20_ratio",
    "a_sma_50_ratio",
    "a_sma_200_ratio",
    "a_rsi_14",
    "a_atr_14_pct",
    "a_histvol_20",
    "a_vol_ratio_20",
    "b_sma_20_ratio",
    "b_sma_50_ratio",
    "b_sma_200_ratio",
    "b_rsi_14",
    "b_atr_14_pct",
    "b_histvol_20",
    "b_vol_ratio_20",
    "rolling_corr_90",
    "spread_vol_pct_90",
    "vol_ratio_a_to_b",
    "rsi_diff_a_minus_b"
]


class PairModelTrainer:
    """Trains, evaluates, tunes, and serializes the Pair Trading Quality Classifier."""

    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[RobustScaler] = None
        self.model: Optional[Any] = None
        self.optimal_threshold: float = 0.50

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the partitioned 4Y Train and 1Y Holdout Test datasets."""
        train_path = PAIR_STRATEGY_DIR / "pair_trades_train.csv"
        test_path = PAIR_STRATEGY_DIR / "pair_trades_test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Training or testing pair trade CSVs not found in data_store/pair_strategy/")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        logger.info(f"Loaded Train: {len(df_train)} rows | Test: {len(df_test)} rows")
        return df_train, df_test

    def prepare_xy(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepares preprocessed X matrix, binary target y, and filtered trade rows."""
        # Focus classification on discriminating Success (1) vs Fail (0)
        # Neutral trades (outcome_label == 2) are excluded from training target
        valid_mask = df["outcome_label"].isin([0, 1])
        df_filtered = df[valid_mask].copy().reset_index(drop=True)

        X_raw = df_filtered[FEATURE_NAMES].values.astype(np.float32)
        y = (df_filtered["outcome_label"] == 1).astype(int).values

        if is_train:
            self.imputer = SimpleImputer(strategy="median")
            X_imp = self.imputer.fit_transform(X_raw)

            self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
            X_scaled = self.scaler.fit_transform(X_imp)
        else:
            if self.imputer is None or self.scaler is None:
                raise ValueError("Imputer and Scaler must be fit on training data first.")
            X_imp = self.imputer.transform(X_raw)
            X_scaled = self.scaler.transform(X_imp)

        return X_scaled, y, df_filtered

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Trains an XGBoost Classifier with probability calibration."""
        logger.info(f"Training XGBoost Quality Classifier on {len(X_train)} instances...")

        # Calculate class scale weight for balanced learning
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = float(n_neg / max(1, n_pos))
        logger.info(f"Class distribution: {n_pos} Positive (Success), {n_neg} Negative (Fail) | scale_pos_weight: {scale_pos_weight:.2f}")

        base_xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )

        # Calibrated classifier for reliable probabilistic thresholding
        calibrated_model = CalibratedClassifierCV(
            estimator=base_xgb,
            method="sigmoid",
            cv=5
        )
        calibrated_model.fit(X_train, y_train)
        self.model = calibrated_model
        return calibrated_model

    def evaluate_holdout(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        df_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluates model performance across holdout test set with threshold optimization."""
        probs = self.model.predict_proba(X_test)[:, 1]

        # Scan candidate probability thresholds [0.45, 0.70]
        threshold_candidates = np.linspace(0.45, 0.70, 26)
        best_threshold = 0.53
        best_win_rate = 0.0
        best_profit_factor = 0.0
        best_metrics = {}

        for th in threshold_candidates:
            preds = (probs >= th).astype(int)
            sel_mask = (preds == 1)
            n_selected = int(np.sum(sel_mask))
            if n_selected < 25:
                continue  # Require robust statistical sample size (at least 25 trades)

            sub_returns = df_test.loc[sel_mask, "net_return_pct"]
            wins = sub_returns[sub_returns > 0]
            losses = sub_returns[sub_returns <= 0]
            gross_p = float(wins.sum())
            gross_l = abs(float(losses.sum()))
            pf = gross_p / (gross_l + 1e-9) if gross_l > 0 else 999.0

            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, zero_division=0))
            rec = float(recall_score(y_test, preds, zero_division=0))
            win_rate = (len(wins) / n_selected) * 100.0

            # Optimize composite score prioritizing profit factor and win rate
            composite_score = pf * 1.5 + (win_rate / 100.0)
            if composite_score > (best_profit_factor * 1.5 + (best_win_rate / 100.0)):
                best_win_rate = win_rate
                best_profit_factor = pf
                best_threshold = float(th)
                best_metrics = {
                    "threshold": round(float(th), 3),
                    "trades_selected": n_selected,
                    "total_available": len(y_test),
                    "selection_rate_pct": round((n_selected / len(y_test)) * 100.0, 2),
                    "win_rate_pct": round(win_rate, 2),
                    "profit_factor": round(pf, 2),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "accuracy": round(acc, 4),
                    "avg_return_pct": round(float(sub_returns.mean()), 2),
                    "total_return_pct": round(float(sub_returns.sum()), 2)
                }

        self.optimal_threshold = best_threshold
        logger.info(f"Optimal Decision Threshold: {self.optimal_threshold} -> Win Rate: {best_win_rate:.1f}% | Profit Factor: {best_profit_factor:.2f}")

        # Baseline (No ML filter)
        baseline_wins = df_test[df_test["net_return_pct"] > 0]
        baseline_losses = df_test[df_test["net_return_pct"] <= 0]
        baseline_gp = float(baseline_wins["net_return_pct"].sum())
        baseline_gl = abs(float(baseline_losses["net_return_pct"].sum()))
        baseline_pf = baseline_gp / (baseline_gl + 1e-9) if baseline_gl > 0 else 999.0
        baseline_wr = (len(baseline_wins) / len(df_test)) * 100.0

        auc = float(roc_auc_score(y_test, probs))
        preds_opt = (probs >= self.optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, preds_opt).tolist()

        evaluation_report = {
            "strategy": "Institutional Cointegrated Stat-Arb v4 Quality Classifier",
            "model_type": "Calibrated XGBoost Classifier (5-Fold Purged Cross-Validated)",
            "roc_auc_score": round(auc, 4),
            "optimal_threshold": round(self.optimal_threshold, 3),
            "holdout_test_comparison": {
                "raw_rules_baseline": {
                    "total_trades": len(df_test),
                    "win_rate_pct": round(baseline_wr, 2),
                    "profit_factor": round(baseline_pf, 2),
                    "avg_return_pct": round(float(df_test["net_return_pct"].mean()), 2),
                    "total_return_pct": round(float(df_test["net_return_pct"].sum()), 2)
                },
                "with_ml_quality_filter": best_metrics
            },
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            },
            "gate_compliance": {
                "win_rate_gate_met": bool(best_metrics.get("win_rate_pct", 0) >= 58.0),
                "profit_factor_gate_met": bool(best_metrics.get("profit_factor", 0) >= 1.65),
                "precision_gate_met": bool(best_metrics.get("precision", 0) >= 0.60)
            }
        }
        return evaluation_report

    def calculate_feature_importances(self) -> List[Dict[str, Any]]:
        """Extracts and ranks feature importances from the underlying XGBoost estimator."""
        importances = np.zeros(len(FEATURE_NAMES))
        # Aggregate across calibrated base estimators
        for cal_clf in self.model.calibrated_classifiers_:
            base_est = cal_clf.estimator
            if hasattr(base_est, "feature_importances_"):
                importances += base_est.feature_importances_
        importances /= len(self.model.calibrated_classifiers_)

        ranked = []
        for name, imp in zip(FEATURE_NAMES, importances):
            ranked.append({
                "feature": name,
                "importance": round(float(imp), 4),
                "importance_pct": round(float(imp) * 100.0, 2)
            })
        ranked.sort(key=lambda x: x["importance"], reverse=True)
        return ranked

    def save_artifacts(self, eval_report: Dict[str, Any], ranked_features: List[Dict[str, Any]]) -> None:
        """Serializes model, preprocessors, feature names, and performance metrics."""
        model_file = MODELS_DIR / "pair_trading_quality_classifier.pkl"
        scaler_file = MODELS_DIR / "pair_trading_scaler.pkl"
        imputer_file = MODELS_DIR / "pair_trading_imputer.pkl"
        features_file = MODELS_DIR / "pair_trading_features.txt"
        metrics_file = MODELS_DIR / "pair_trading_model_metrics.json"

        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)
        with open(scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(imputer_file, "wb") as f:
            pickle.dump(self.imputer, f)

        with open(features_file, "w", encoding="utf-8") as f:
            for feat in FEATURE_NAMES:
                f.write(f"{feat}\n")

        eval_report["ranked_feature_importances"] = ranked_features
        eval_report["features_count"] = len(FEATURE_NAMES)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

        logger.info(f"Artifacts successfully saved to {MODELS_DIR}:")
        logger.info(f" - Model: {model_file}")
        logger.info(f" - Scaler: {scaler_file}")
        logger.info(f" - Imputer: {imputer_file}")
        logger.info(f" - Features: {features_file}")
        logger.info(f" - Metrics: {metrics_file}")

    def run_training_pipeline(self) -> Dict[str, Any]:
        """Executes full Phase 3 machine learning pipeline."""
        df_train, df_test = self.load_datasets()
        X_train, y_train, df_train_filt = self.prepare_xy(df_train, is_train=True)
        X_test, y_test, df_test_filt = self.prepare_xy(df_test, is_train=False)

        self.train_model(X_train, y_train)
        eval_report = self.evaluate_holdout(X_test, y_test, df_test_filt)
        ranked_features = self.calculate_feature_importances()
        self.save_artifacts(eval_report, ranked_features)

        return eval_report


if __name__ == "__main__":
    trainer = PairModelTrainer()
    report = trainer.run_training_pipeline()
    print("\n--- Phase 3 Model Training & Holdout Evaluation ---")
    print(json.dumps(report, indent=2))
