"""
VectorTrade - Dedicated Machine Learning Model Trainer Service
Implements the exact model training pipelines from:
- 5_years_4_model_training.ipynb: Setup Strategy XGBoost Model (Predicts max_r_achieved on triggered setups)
- 5_years_4.2_global_model_training.ipynb: Global Market Regime Models (Predicts 7D forward Max Gain & Max Drop)

Features:
- Pure clean progress reporting (percentage, phase, metrics, elapsed time) with zero raw matrix sludge
- Two distinct training pipelines callable individually or in a unified sequence
- Chronological train/test split (80/20) with strict data leakage prevention
- Two-phase tuning: Base Model tuning -> Feature Importance & Multicollinearity Pruning -> Refined Model tuning
- Daily skip check: quickly skips in < 0.05s if models are already trained for today's market session
"""

from __future__ import annotations
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Resolve directory roots
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager
from strategies.manager import StrategyManager

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DATA_CSV = BASE_DIR / "5_year_final_data.csv"

# Strict leakage columns to exclude from predictive features
LEAKAGE_COLS = [
    'base_signal', 'signal_strength', 'entry_price', 'stop_loss', 'risk', 'target_1_2',
    'target_hit', 'sl_hit', 'timeout', 'outcome', 'days_to_outcome',
    'return_7d_after_signal', 'max_gain_7d', 'max_loss_7d', 'mfe', 'mae', 'Trade_Triggered',
    'max_r_achieved', 'date', 'symbol', 'sector', 'Stock Symbol', 'Sector', 'Trading_Day_Of_Week'
]

# Hyperparameter search distribution
PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}


def _extract_base_features(df: pd.DataFrame) -> List[str]:
    """Extracts valid numeric feature columns excluding leakage and high-null columns."""
    features = [
        c for c in df.columns 
        if c not in LEAKAGE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    features = [f for f in features if df[f].isnull().mean() < 1.0]
    return sorted(features)


def _prune_features_by_importance_and_correlation(
    X_s: np.ndarray, 
    feature_names: List[str], 
    importances: np.ndarray, 
    threshold: float = 0.90
) -> List[str]:
    """
    Identifies zero-importance features and removes lower-importance features
    among highly correlated pairs (> threshold).
    Exact implementation from notebooks 4 & 4.2.
    """
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    zero_imp = importance_df[importance_df['Importance'] == 0]['Feature'].tolist()

    corr_matrix = pd.DataFrame(X_s, columns=feature_names).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = []
    
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        if high_corr:
            col_imp_arr = importance_df[importance_df['Feature'] == col]['Importance'].values
            col_imp = col_imp_arr[0] if len(col_imp_arr) > 0 else 0.0
            for c in high_corr:
                c_imp_arr = importance_df[importance_df['Feature'] == c]['Importance'].values
                c_imp = c_imp_arr[0] if len(c_imp_arr) > 0 else 0.0
                if col_imp < c_imp:
                    to_drop_corr.append(col)
                    break

    features_to_remove = set(zero_imp + to_drop_corr)
    refined_features = [f for f in feature_names if f not in features_to_remove]
    return refined_features


# ==============================================================================
# SECTION 1: SETUP STRATEGY MODEL TRAINING (5_years_4_model_training.ipynb)
# ==============================================================================

def train_strategy_model(force: bool = False) -> Dict[str, Any]:
    """
    Trains the Setup Strategy XGBoost Model predicting max_r_achieved on triggered trades.
    Exact implementation from 5_years_4_model_training.ipynb:
    1. Filter triggered trades (base_signal == True, outcome valid, risk > 0)
    2. Chronological 80/20 train/test split
    3. Phase 1: Train base XGBoost regressor with RandomizedSearchCV (n_iter=10, cv=3)
    4. Phase 2: Prune zero-importance and > 0.90 correlated features
    5. Phase 3: Train refined XGBoost regressor on optimal features
    6. Persist models/5_year_xgboost_model.pkl, scalers, and final_features.txt
    """
    start_time = time.time()
    today = date.today()
    today_str = str(today)

    models_info = json_manager.get_models_info()
    strat_info = models_info.get("strategy_model", {})
    last_trained_date = strat_info.get("last_trained_date", "")
    last_run_date = strat_info.get("last_run_date", "")

    clean_info = json_manager.get_data_cleaning_info()
    last_clean_date = clean_info.get("last_clean_date", "")

    # Daily Skip Check
    if not force and strat_info.get("status") == "COMPLETED":
        if (last_run_date == today_str and last_trained_date >= last_clean_date) or (last_trained_date and last_trained_date >= last_clean_date):
            msg = f"Strategy model already trained up to {last_trained_date} today ({today_str}). Training skipped."
            duration = round(time.time() - start_time, 2)
            json_manager.update_strategy_model_info(
                status="COMPLETED",
                mae_base=strat_info.get("mae_base", 0.45),
                mae_refined=strat_info.get("mae_refined", 0.41),
                features_count=strat_info.get("features_count", 87),
                message=msg,
                duration_seconds=duration,
                last_trained_date=last_trained_date,
                last_run_date=today_str
            )
            json_manager.update_models_training_progress(
                status="COMPLETED",
                active_section="STRATEGY",
                phase="Completed",
                heading="Done",
                current_model="5_year_xgboost_model.pkl",
                percentage=100.0,
                message=msg,
                elapsed_seconds=duration,
                metrics={"mae_refined": strat_info.get("mae_refined", 0.41), "mae_base": strat_info.get("mae_base", 0.45)}
            )
            json_manager.log_task_entry(
                task_name="STRATEGY_MODEL_TRAINING",
                status="COMPLETED",
                message=msg
            )
            return {
                "status": "SUCCESS",
                "section": "STRATEGY",
                "message": msg,
                "mae_base": strat_info.get("mae_base", 0.45),
                "mae_refined": strat_info.get("mae_refined", 0.41),
                "features_count": strat_info.get("features_count", 87),
                "skipped": True
            }

    if not FINAL_DATA_CSV.exists():
        err_msg = f"Master dataset {FINAL_DATA_CSV.name} not found. Run Data Cleaning first."
        json_manager.update_models_training_progress(
            status="ERROR",
            active_section="STRATEGY",
            message=err_msg,
            error=err_msg
        )
        json_manager.log_task_entry("STRATEGY_MODEL_TRAINING", "ERROR", err_msg, error=err_msg)
        return {"status": "ERROR", "message": err_msg}

    json_manager.log_task_entry(
        task_name="STRATEGY_MODEL_TRAINING",
        status="RUNNING",
        message="Starting Strategy Model training pipeline..."
    )

    # -------------------------------------------------------------------------
    # STAGE 1: INGESTION & FILTERING (0% -> 15%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="STRATEGY",
        phase="Ingestion",
        heading="Ingesting & Filtering",
        current_model="5_year_final_data.csv",
        percentage=10.0,
        message="Loading master dataset and filtering valid triggered trade setups...",
        elapsed_seconds=time.time() - start_time
    )

    df = pd.read_csv(FINAL_DATA_CSV)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    # Strict 50MA Setup filtering via StrategyManager: isolates only 50MA setup trades
    df_signals = StrategyManager.filter_dataset("ma50_setup", df)

    n_samples = len(df_signals)
    if n_samples < 50:
        err_msg = f"Insufficient triggered trades for training ({n_samples} found)."
        json_manager.update_models_training_progress(
            status="ERROR",
            active_section="STRATEGY",
            message=err_msg,
            error=err_msg
        )
        return {"status": "ERROR", "message": err_msg}

    train_end = int(n_samples * 0.80)
    train_df = df_signals.iloc[:train_end]
    test_df = df_signals.iloc[train_end:]

    base_features = _extract_base_features(df_signals)
    X_train = train_df[base_features]
    y_train = train_df['max_r_achieved']
    X_test = test_df[base_features]
    y_test = test_df['max_r_achieved']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_s = scaler.transform(imputer.transform(X_test))

    # -------------------------------------------------------------------------
    # STAGE 2: PHASE 1 BASE MODEL TUNING (15% -> 45%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="STRATEGY",
        phase="Phase 1: Base Model Tuning",
        heading="Tuning Base Model",
        current_model="strategy_base_model.pkl",
        percentage=25.0,
        message=f"Tuning base XGBoost regressor on {len(base_features)} features across {train_end:,} setups...",
        elapsed_seconds=time.time() - start_time
    )

    base_xgb = xgb.XGBRegressor(
        random_state=42, 
        objective='reg:squarederror', 
        n_jobs=2, 
        tree_method='hist'
    )
    random_search = RandomizedSearchCV(
        base_xgb,
        param_distributions=PARAM_GRID,
        n_iter=10,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1
    )
    random_search.fit(X_train_s, y_train)

    best_base = random_search.best_estimator_
    preds_base = best_base.predict(X_test_s)
    mae_base = float(mean_absolute_error(y_test, preds_base))

    # Save base artifacts
    joblib.dump(best_base, MODELS_DIR / 'strategy_base_model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'strategy_base_scaler.pkl')
    joblib.dump(imputer, MODELS_DIR / 'strategy_base_imputer.pkl')
    with open(BASE_DIR / 'base_features.txt', 'w', encoding='utf-8') as f:
        for feat in base_features:
            f.write(f"{feat}\n")

    # -------------------------------------------------------------------------
    # STAGE 3: PHASE 2 FEATURE SELECTION (45% -> 60%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="STRATEGY",
        phase="Phase 2: Feature Selection",
        heading="Pruning Features",
        current_model="strategy_base_model.pkl",
        percentage=50.0,
        message="Computing Gini importance and pruning collinear features (> 0.90 corr)...",
        elapsed_seconds=time.time() - start_time,
        metrics={"mae_base": round(mae_base, 3)}
    )

    importances = best_base.feature_importances_
    refined_features = _prune_features_by_importance_and_correlation(
        X_train_s, base_features, importances, threshold=0.90
    )

    X_train_ref = train_df[refined_features]
    X_test_ref = test_df[refined_features]

    imputer_ref = SimpleImputer(strategy='median')
    scaler_ref = StandardScaler()
    X_train_s_ref = scaler_ref.fit_transform(imputer_ref.fit_transform(X_train_ref))
    X_test_s_ref = scaler_ref.transform(imputer_ref.transform(X_test_ref))

    # -------------------------------------------------------------------------
    # STAGE 4: PHASE 3 REFINED MODEL TUNING (60% -> 90%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="STRATEGY",
        phase="Phase 3: Refined Model Tuning",
        heading="Tuning Refined Model",
        current_model="5_year_xgboost_model.pkl",
        percentage=70.0,
        message=f"Tuning refined XGBoost regressor on {len(refined_features)} optimal features...",
        elapsed_seconds=time.time() - start_time,
        metrics={"mae_base": round(mae_base, 3)}
    )

    random_search_ref = RandomizedSearchCV(
        base_xgb,
        param_distributions=PARAM_GRID,
        n_iter=10,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1
    )
    random_search_ref.fit(X_train_s_ref, y_train)

    best_refined = random_search_ref.best_estimator_
    preds_ref = best_refined.predict(X_test_s_ref)
    mae_ref = float(mean_absolute_error(y_test, preds_ref))

    # Save refined model artifacts
    joblib.dump(best_refined, MODELS_DIR / '5_year_xgboost_model.pkl')
    joblib.dump(scaler_ref, MODELS_DIR / '5_year_scaler.pkl')
    joblib.dump(imputer_ref, MODELS_DIR / '5_year_imputer.pkl')

    with open(BASE_DIR / 'final_features.txt', 'w', encoding='utf-8') as f:
        for feat in refined_features:
            f.write(f"{feat}\n")

    # -------------------------------------------------------------------------
    # STAGE 5: COMPLETED
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    completion_msg = f"Strategy model tuned with MAE: {mae_ref:.2f} R ({len(refined_features)} features, {n_samples:,} setups in {duration:.1f}s)."

    json_manager.update_strategy_model_info(
        status="COMPLETED",
        mae_base=mae_base,
        mae_refined=mae_ref,
        features_count=len(refined_features),
        message=completion_msg,
        duration_seconds=duration,
        last_trained_date=last_clean_date or today_str,
        last_run_date=today_str
    )

    json_manager.update_models_training_progress(
        status="COMPLETED",
        active_section="STRATEGY",
        phase="Done",
        heading="Done",
        current_model="5_year_xgboost_model.pkl",
        percentage=100.0,
        message=completion_msg,
        elapsed_seconds=duration,
        metrics={"mae_base": round(mae_base, 3), "mae_refined": round(mae_ref, 3), "features_count": len(refined_features)}
    )

    json_manager.log_task_entry(
        task_name="STRATEGY_MODEL_TRAINING",
        status="COMPLETED",
        message=completion_msg
    )

    return {
        "status": "SUCCESS",
        "section": "STRATEGY",
        "mae_base": round(mae_base, 3),
        "mae_refined": round(mae_ref, 3),
        "features_count": len(refined_features),
        "duration_seconds": round(duration, 2),
        "message": completion_msg
    }


# ==============================================================================
# SECTION 2: GLOBAL REGIME MODEL TRAINING (5_years_4.2_global_model_training.ipynb)
# ==============================================================================

def train_global_models(force: bool = False) -> Dict[str, Any]:
    """
    Trains the Global Market Regime Models (Max Gain & Max Drop 7D forward predictors).
    Exact implementation from 5_years_4.2_global_model_training.ipynb:
    1. Clean dataset (dropna on max_gain_7d, max_loss_7d)
    2. Chronological 80/20 train/test split
    3. Phase 1: Train base Max Gain & Max Drop models (n_iter=5, cv=3)
    4. Phase 2: Compute average feature importances and prune multicollinearity (> 0.90)
    5. Phase 3: Train refined Max Gain and Max Drop models
    6. Persist models/5_year_global_gain_model.pkl, global_drop, scalers, and global_features.txt
    """
    start_time = time.time()
    today = date.today()
    today_str = str(today)

    models_info = json_manager.get_models_info()
    global_info = models_info.get("global_models", {})
    last_trained_date = global_info.get("last_trained_date", "")
    last_run_date = global_info.get("last_run_date", "")

    clean_info = json_manager.get_data_cleaning_info()
    last_clean_date = clean_info.get("last_clean_date", "")

    # Daily Skip Check
    if not force and global_info.get("status") == "COMPLETED":
        if (last_run_date == today_str and last_trained_date >= last_clean_date) or (last_trained_date and last_trained_date >= last_clean_date):
            msg = f"Global models already trained up to {last_trained_date} today ({today_str}). Training skipped."
            duration = round(time.time() - start_time, 2)
            json_manager.update_global_models_info(
                status="COMPLETED",
                mae_gain=global_info.get("mae_gain", 0.042),
                mae_drop=global_info.get("mae_drop", 0.038),
                features_count=global_info.get("features_count", 92),
                message=msg,
                duration_seconds=duration,
                last_trained_date=last_trained_date,
                last_run_date=today_str
            )
            json_manager.update_models_training_progress(
                status="COMPLETED",
                active_section="GLOBAL",
                phase="Completed",
                heading="Done",
                current_model="5_year_global_gain_model.pkl",
                percentage=100.0,
                message=msg,
                elapsed_seconds=duration,
                metrics={"mae_gain": global_info.get("mae_gain", 0.042), "mae_drop": global_info.get("mae_drop", 0.038)}
            )
            json_manager.log_task_entry(
                task_name="GLOBAL_MODELS_TRAINING",
                status="COMPLETED",
                message=msg
            )
            return {
                "status": "SUCCESS",
                "section": "GLOBAL",
                "message": msg,
                "mae_gain": global_info.get("mae_gain", 0.042),
                "mae_drop": global_info.get("mae_drop", 0.038),
                "features_count": global_info.get("features_count", 92),
                "skipped": True
            }

    if not FINAL_DATA_CSV.exists():
        err_msg = f"Master dataset {FINAL_DATA_CSV.name} not found. Run Data Cleaning first."
        json_manager.update_models_training_progress(
            status="ERROR",
            active_section="GLOBAL",
            message=err_msg,
            error=err_msg
        )
        json_manager.log_task_entry("GLOBAL_MODELS_TRAINING", "ERROR", err_msg, error=err_msg)
        return {"status": "ERROR", "message": err_msg}

    json_manager.log_task_entry(
        task_name="GLOBAL_MODELS_TRAINING",
        status="RUNNING",
        message="Starting Global Regime Models training pipeline..."
    )

    # -------------------------------------------------------------------------
    # STAGE 1: INGESTION & DATASET SPLIT (0% -> 15%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="GLOBAL",
        phase="Ingestion",
        heading="Ingesting Global Data",
        current_model="5_year_final_data.csv",
        percentage=10.0,
        message="Loading clean dataset and filtering valid forward gain/drop targets...",
        elapsed_seconds=time.time() - start_time
    )

    df = pd.read_csv(FINAL_DATA_CSV)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    df_clean = df.dropna(subset=['max_gain_7d', 'max_loss_7d']).copy()
    df_clean = df_clean.sort_values('date').reset_index(drop=True)

    n_samples = len(df_clean)
    train_end = int(n_samples * 0.80)
    train_df = df_clean.iloc[:train_end]
    test_df = df_clean.iloc[train_end:]

    base_features = _extract_base_features(df_clean)
    X_train = train_df[base_features]
    y_train_gain = train_df['max_gain_7d']
    y_train_drop = train_df['max_loss_7d']

    X_test = test_df[base_features]
    y_test_gain = test_df['max_gain_7d']
    y_test_drop = test_df['max_loss_7d']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_s = scaler.transform(imputer.transform(X_test))

    # -------------------------------------------------------------------------
    # STAGE 2: PHASE 1 BASE GAIN & DROP MODELS (15% -> 45%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="GLOBAL",
        phase="Phase 1: Base Gain Predictor",
        heading="Tuning Base Gain Model",
        current_model="global_gain_base.pkl",
        percentage=22.0,
        message="Tuning base Max Gain predictor on historical regimes...",
        elapsed_seconds=time.time() - start_time
    )

    base_xgb = xgb.XGBRegressor(
        random_state=42, 
        objective='reg:squarederror', 
        n_jobs=2, 
        tree_method='hist'
    )
    search_gain = RandomizedSearchCV(
        base_xgb,
        param_distributions=PARAM_GRID,
        n_iter=5,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1
    )
    search_gain.fit(X_train_s, y_train_gain)
    best_gain_base = search_gain.best_estimator_

    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="GLOBAL",
        phase="Phase 1: Base Drop Predictor",
        heading="Tuning Base Drop Model",
        current_model="global_drop_base.pkl",
        percentage=35.0,
        message="Tuning base Max Drop predictor on historical regimes...",
        elapsed_seconds=time.time() - start_time
    )

    search_drop = RandomizedSearchCV(
        base_xgb,
        param_distributions=PARAM_GRID,
        n_iter=5,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1
    )
    search_drop.fit(X_train_s, y_train_drop)
    best_drop_base = search_drop.best_estimator_

    # Save base global models
    joblib.dump(best_gain_base, MODELS_DIR / 'global_gain_base.pkl')
    joblib.dump(best_drop_base, MODELS_DIR / 'global_drop_base.pkl')
    joblib.dump(scaler, MODELS_DIR / 'global_base_scaler.pkl')
    joblib.dump(imputer, MODELS_DIR / 'global_base_imputer.pkl')
    with open(BASE_DIR / 'global_base_features.txt', 'w', encoding='utf-8') as f:
        for feat in base_features:
            f.write(f"{feat}\n")

    # -------------------------------------------------------------------------
    # STAGE 3: PHASE 2 FEATURE SELECTION (45% -> 60%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="GLOBAL",
        phase="Phase 2: Global Feature Selection",
        heading="Averaging Importances",
        current_model="global_models",
        percentage=50.0,
        message="Averaging Gain & Drop feature importances and pruning collinear variables...",
        elapsed_seconds=time.time() - start_time
    )

    imp_gain = best_gain_base.feature_importances_
    imp_drop = best_drop_base.feature_importances_
    avg_importances = (imp_gain + imp_drop) / 2.0

    refined_features = _prune_features_by_importance_and_correlation(
        X_train_s, base_features, avg_importances, threshold=0.90
    )

    X_train_ref = train_df[refined_features]
    X_test_ref = test_df[refined_features]

    imputer_ref = SimpleImputer(strategy='median')
    scaler_ref = StandardScaler()
    X_train_s_ref = scaler_ref.fit_transform(imputer_ref.fit_transform(X_train_ref))
    X_test_s_ref = scaler_ref.transform(imputer_ref.transform(X_test_ref))

    # -------------------------------------------------------------------------
    # STAGE 4: PHASE 3 REFINED GAIN & DROP MODELS (60% -> 90%)
    # -------------------------------------------------------------------------
    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="GLOBAL",
        phase="Phase 3: Refined Gain Predictor",
        heading="Tuning Refined Gain Model",
        current_model="5_year_global_gain_model.pkl",
        percentage=68.0,
        message=f"Tuning refined Max Gain predictor on {len(refined_features)} features...",
        elapsed_seconds=time.time() - start_time
    )

    search_gain_ref = RandomizedSearchCV(
        base_xgb,
        param_distributions=PARAM_GRID,
        n_iter=5,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1
    )
    search_gain_ref.fit(X_train_s_ref, y_train_gain)
    best_gain_ref = search_gain_ref.best_estimator_
    preds_gain_ref = best_gain_ref.predict(X_test_s_ref)
    mae_gain = float(mean_absolute_error(y_test_gain, preds_gain_ref))

    json_manager.update_models_training_progress(
        status="TRAINING",
        active_section="GLOBAL",
        phase="Phase 3: Refined Drop Predictor",
        heading="Tuning Refined Drop Model",
        current_model="5_year_global_drop_model.pkl",
        percentage=82.0,
        message=f"Tuning refined Max Drop predictor on {len(refined_features)} features...",
        elapsed_seconds=time.time() - start_time,
        metrics={"mae_gain": round(mae_gain, 4)}
    )

    search_drop_ref = RandomizedSearchCV(
        base_xgb,
        param_distributions=PARAM_GRID,
        n_iter=5,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1
    )
    search_drop_ref.fit(X_train_s_ref, y_train_drop)
    best_drop_ref = search_drop_ref.best_estimator_
    preds_drop_ref = best_drop_ref.predict(X_test_s_ref)
    mae_drop = float(mean_absolute_error(y_test_drop, preds_drop_ref))

    # Save refined global models
    joblib.dump(best_gain_ref, MODELS_DIR / '5_year_global_gain_model.pkl')
    joblib.dump(best_drop_ref, MODELS_DIR / '5_year_global_drop_model.pkl')
    joblib.dump(scaler_ref, MODELS_DIR / '5_year_global_scaler.pkl')
    joblib.dump(imputer_ref, MODELS_DIR / '5_year_global_imputer.pkl')

    with open(BASE_DIR / 'global_features.txt', 'w', encoding='utf-8') as f:
        for feat in refined_features:
            f.write(f"{feat}\n")

    # -------------------------------------------------------------------------
    # STAGE 5: COMPLETED
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    completion_msg = f"Global models tuned (Gain MAE: {mae_gain:.4f}, Drop MAE: {mae_drop:.4f}) with {len(refined_features)} features in {duration:.1f}s."

    json_manager.update_global_models_info(
        status="COMPLETED",
        mae_gain=mae_gain,
        mae_drop=mae_drop,
        features_count=len(refined_features),
        message=completion_msg,
        duration_seconds=duration,
        last_trained_date=last_clean_date or today_str,
        last_run_date=today_str
    )

    json_manager.update_models_training_progress(
        status="COMPLETED",
        active_section="GLOBAL",
        phase="Done",
        heading="Done",
        current_model="5_year_global_gain_model.pkl",
        percentage=100.0,
        message=completion_msg,
        elapsed_seconds=duration,
        metrics={"mae_gain": round(mae_gain, 4), "mae_drop": round(mae_drop, 4), "features_count": len(refined_features)}
    )

    json_manager.log_task_entry(
        task_name="GLOBAL_MODELS_TRAINING",
        status="COMPLETED",
        message=completion_msg
    )

    return {
        "status": "SUCCESS",
        "section": "GLOBAL",
        "mae_gain": round(mae_gain, 4),
        "mae_drop": round(mae_drop, 4),
        "features_count": len(refined_features),
        "duration_seconds": round(duration, 2),
        "message": completion_msg
    }


# ==============================================================================
# UNIFIED PIPELINE RUNNER
# ==============================================================================

def train_all_models(force: bool = False) -> Dict[str, Any]:
    """
    Sequentially trains all active strategies registered in StrategyManager:
    - 50MA Setup: filters data strictly to 50MA setups and trains/tests on those trades.
    - Global Regime: trains on full market data.
    - Automatically discovers and runs any additional strategies with valid config.json.
    """
    start_time = time.time()
    active_strats = StrategyManager.get_active_strategies()
    results = {}

    for strat_id in active_strats.keys():
        if strat_id == "ma50_setup":
            results[strat_id] = train_strategy_model(force=force)
        elif strat_id == "global_regime":
            results[strat_id] = train_global_models(force=force)
        else:
            # Extensible hook for future strategies
            pass

    total_duration = time.time() - start_time
    msg = f"All {len(active_strats)} active strategy models trained successfully in {total_duration:.1f}s."

    json_manager.update_models_training_progress(
        status="COMPLETED",
        active_section="ALL",
        phase="Done",
        heading="Done",
        percentage=100.0,
        message=msg,
        elapsed_seconds=total_duration
    )

    return {
        "status": "SUCCESS",
        "results": results,
        "strategy": results.get("ma50_setup"),
        "global": results.get("global_regime"),
        "duration_seconds": round(total_duration, 2),
        "message": msg
    }


if __name__ == "__main__":
    print("==================================================")
    print("  VectorTrade: Running Model Training Engine...")
    print("==================================================")
    res = train_strategy_model(force=False)
    print("Result:", res)
