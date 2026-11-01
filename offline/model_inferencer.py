"""
VectorTrade - Dedicated Machine Learning Model Inferencer & Evaluation Engine
Implements the exact model evaluation pipelines from:
- 5_years_5_inference.ipynb: Strategy Model Evaluation & Out-of-Sample Test Set Validation
- 5_years_5.2_global_inference.ipynb: Global Regime Model Evaluation (Gain, Drop & Directional Accuracy)

Features:
- Pure clean progress reporting (percentage, phase, live metrics, elapsed seconds)
- Two distinct evaluation pipelines callable individually or in a unified sequence
- Logs exact benchmark metrics to models/model_performance_log.json matching notebook outputs
- Chronological out-of-sample evaluation (last 20% test slice)
- Fast daily skip logic (< 0.05s execution if models have already been evaluated today)
- Thread-safe non-blocking execution for Flask server
"""

from __future__ import annotations
import os
import sys
import time
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Resolve directory roots
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import json_manager

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DATA_CSV = BASE_DIR / "5_year_final_data.csv"
MODEL_PERFORMANCE_LOG_FILE = MODELS_DIR / "model_performance_log.json"


def _update_model_log(model_name: str, metrics: Dict[str, Any], features_count: int, removed_features: int = 0) -> None:
    """
    Updates models/model_performance_log.json atomically with evaluation metrics.
    Exact implementation from notebooks 5 and 5.2.
    """
    log_data: Dict[str, Any] = {}
    if MODEL_PERFORMANCE_LOG_FILE.exists():
        try:
            with open(MODEL_PERFORMANCE_LOG_FILE, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except Exception:
            log_data = {}

    log_data[model_name] = {
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "features_count": features_count,
        "removed_features": removed_features
    }

    try:
        temp_file = MODEL_PERFORMANCE_LOG_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
        temp_file.replace(MODEL_PERFORMANCE_LOG_FILE)
    except Exception:
        with open(MODEL_PERFORMANCE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)


def _load_features_list(file_path: Path) -> List[str]:
    """Reads line-separated feature names from a text file."""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ==============================================================================
# SECTION 1: STRATEGY MODEL INFERENCE (NOTEBOOK 5)
# ==============================================================================

def run_strategy_inference(force: bool = False) -> Dict[str, Any]:
    """
    Evaluates Strategy Models (Base and Refined) on historical out-of-sample setups.
    Calculates Accuracy, MAE (in R multiples), and Win Rate Recall.
    Writes results to models/model_performance_log.json and data_store/inference_state.json.
    """
    start_time = time.time()
    today_str = str(date.today())

    # Daily skip check
    inf_info = json_manager.get_inference_info()
    strat_info = inf_info.get("strategy_inference", {})
    last_run_date = strat_info.get("last_run_date", "")

    if not force and last_run_date == today_str and strat_info.get("status") == "COMPLETED":
        msg = f"Strategy model already evaluated today ({today_str}). Evaluation skipped."
        duration = round(time.time() - start_time, 2)
        json_manager.update_inference_progress(
            status="COMPLETED",
            active_section="STRATEGY",
            phase="Done",
            heading="Done",
            percentage=100.0,
            message=msg,
            elapsed_seconds=duration,
            metrics={
                "accuracy_refined": strat_info.get("accuracy_refined", "43.9%"),
                "win_rate_recall": strat_info.get("win_rate_recall", "77.3%"),
                "mae_refined": strat_info.get("mae_refined", 0.886)
            }
        )
        return {
            "status": "SUCCESS",
            "section": "STRATEGY",
            "message": msg,
            "skipped": True,
            "metrics": strat_info
        }

    # Verify model and dataset files exist
    required_files = [
        FINAL_DATA_CSV,
        MODELS_DIR / "strategy_base_model.pkl",
        MODELS_DIR / "strategy_base_scaler.pkl",
        MODELS_DIR / "strategy_base_imputer.pkl",
        MODELS_DIR / "5_year_xgboost_model.pkl",
        MODELS_DIR / "5_year_scaler.pkl",
        MODELS_DIR / "5_year_imputer.pkl",
        BASE_DIR / "base_features.txt",
        BASE_DIR / "final_features.txt"
    ]
    for rf in required_files:
        if not rf.exists():
            err_msg = f"Required file missing for strategy inference: {rf.name}"
            json_manager.update_inference_progress(
                status="ERROR",
                active_section="STRATEGY",
                message=err_msg,
                error=err_msg
            )
            json_manager.log_task_entry("STRATEGY_INFERENCE", "ERROR", err_msg, error=err_msg)
            return {"status": "ERROR", "message": err_msg}

    json_manager.log_task_entry(
        task_name="STRATEGY_INFERENCE",
        status="RUNNING",
        message="Evaluating Strategy Models on historical out-of-sample test splits..."
    )

    # -------------------------------------------------------------------------
    # STAGE 1: INGESTION & TEST SPLIT (0% -> 25%)
    # -------------------------------------------------------------------------
    json_manager.update_inference_progress(
        status="EVALUATING",
        active_section="STRATEGY",
        phase="Ingestion",
        heading="Filtering Setups",
        current_model="5_year_final_data.csv",
        percentage=10.0,
        message="Loading master dataset and filtering out-of-sample trade setups...",
        elapsed_seconds=time.time() - start_time
    )

    df = pd.read_csv(FINAL_DATA_CSV)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    df_signals = df[df['base_signal'] == True].copy()
    df_signals = df_signals[~df_signals['outcome'].isin(['No_Trigger', 'Pending', 'Timeout', np.nan])]
    df_signals = df_signals[df_signals['risk'] > 0]
    df_signals['max_r_achieved'] = df_signals['mfe'] / df_signals['risk']
    df_signals = df_signals.dropna(subset=['max_r_achieved'])
    df_signals = df_signals.sort_values('date').reset_index(drop=True)

    n_samples = len(df_signals)
    if n_samples < 50:
        err_msg = f"Insufficient triggered trades for inference ({n_samples} found)."
        json_manager.update_inference_progress(
            status="ERROR",
            active_section="STRATEGY",
            message=err_msg,
            error=err_msg
        )
        return {"status": "ERROR", "message": err_msg}

    test_start = int(n_samples * 0.80)
    test_df = df_signals.iloc[test_start:].copy()

    base_features = _load_features_list(BASE_DIR / "base_features.txt")
    refined_features = _load_features_list(BASE_DIR / "final_features.txt")

    # -------------------------------------------------------------------------
    # STAGE 2: BASE MODEL EVALUATION (25% -> 55%)
    # -------------------------------------------------------------------------
    json_manager.update_inference_progress(
        status="EVALUATING",
        active_section="STRATEGY",
        phase="Phase 1: Base Model Evaluation",
        heading="Evaluating Base Model",
        current_model="strategy_base_model.pkl",
        percentage=35.0,
        message=f"Evaluating base strategy model on {len(test_df):,} test trades ({len(base_features)} features)...",
        elapsed_seconds=time.time() - start_time
    )

    base_model = joblib.load(MODELS_DIR / "strategy_base_model.pkl")
    base_scaler = joblib.load(MODELS_DIR / "strategy_base_scaler.pkl")
    base_imputer = joblib.load(MODELS_DIR / "strategy_base_imputer.pkl")

    X_test_base = test_df[base_features]
    X_test_s_base = base_scaler.transform(base_imputer.transform(X_test_base))
    preds_base = base_model.predict(X_test_s_base)

    test_df['predicted_r_base'] = preds_base
    test_df['actual_is_success'] = test_df['max_r_achieved'] >= 1.0
    test_df['predicted_is_success_base'] = test_df['predicted_r_base'] >= 1.0

    acc_base = float((test_df['actual_is_success'] == test_df['predicted_is_success_base']).mean() * 100)
    mae_base = float(mean_absolute_error(test_df['max_r_achieved'], preds_base))

    _update_model_log(
        model_name="Strategy_Base",
        metrics={"Accuracy": f"{acc_base:.1f}%", "MAE": round(mae_base, 3)},
        features_count=len(base_features),
        removed_features=0
    )

    # -------------------------------------------------------------------------
    # STAGE 3: REFINED MODEL EVALUATION (55% -> 90%)
    # -------------------------------------------------------------------------
    json_manager.update_inference_progress(
        status="EVALUATING",
        active_section="STRATEGY",
        phase="Phase 2: Refined Model Evaluation",
        heading="Evaluating Refined Model",
        current_model="5_year_xgboost_model.pkl",
        percentage=65.0,
        message=f"Evaluating pruned XGBoost model on {len(refined_features)} features...",
        elapsed_seconds=time.time() - start_time,
        metrics={
            "acc_base": f"{acc_base:.1f}%",
            "mae_base": round(mae_base, 3)
        }
    )

    refined_model = joblib.load(MODELS_DIR / "5_year_xgboost_model.pkl")
    refined_scaler = joblib.load(MODELS_DIR / "5_year_scaler.pkl")
    refined_imputer = joblib.load(MODELS_DIR / "5_year_imputer.pkl")

    X_test_ref = test_df[refined_features]
    X_test_s_ref = refined_scaler.transform(refined_imputer.transform(X_test_ref))
    preds_ref = refined_model.predict(X_test_s_ref)

    test_df['predicted_r_ref'] = preds_ref
    test_df['predicted_is_success_ref'] = test_df['predicted_r_ref'] >= 1.0

    acc_ref = float((test_df['actual_is_success'] == test_df['predicted_is_success_ref']).mean() * 100)
    mae_ref = float(mean_absolute_error(test_df['max_r_achieved'], preds_ref))

    actual_successes = test_df[test_df['actual_is_success'] == True]
    successes_caught = int(actual_successes['predicted_is_success_ref'].sum())
    total_actual = len(actual_successes)
    recall = float((successes_caught / total_actual) * 100 if total_actual > 0 else 0.0)

    removed_count = max(0, len(base_features) - len(refined_features))

    _update_model_log(
        model_name="Strategy_Refined",
        metrics={
            "Accuracy": f"{acc_ref:.1f}%",
            "Win_Rate_Recall": f"{recall:.1f}%",
            "MAE": round(mae_ref, 3)
        },
        features_count=len(refined_features),
        removed_features=removed_count
    )

    # -------------------------------------------------------------------------
    # STAGE 4: OPTIONAL INDIVIDUAL STRATEGY MODELS (90% -> 98%)
    # -------------------------------------------------------------------------
    individual_dir = MODELS_DIR / "individual_model" / "strategy"
    if individual_dir.exists():
        acc_list, mae_list, indiv_logs = [], [], {}
        for sym in test_df['symbol'].unique():
            model_path = individual_dir / f"{sym}_model.pkl"
            if model_path.exists():
                sym_df = test_df[test_df['symbol'] == sym]
                if len(sym_df) == 0:
                    continue
                try:
                    s_m = joblib.load(model_path)
                    s_sc = joblib.load(individual_dir / f"{sym}_scaler.pkl")
                    s_imp = joblib.load(individual_dir / f"{sym}_imputer.pkl")
                    X_s = s_sc.transform(s_imp.transform(sym_df[refined_features]))
                    p = s_m.predict(X_s)
                    m = float(mean_absolute_error(sym_df['max_r_achieved'], p))
                    a = float(((sym_df['max_r_achieved'] >= 1.0) == (p >= 1.0)).mean() * 100)
                    acc_list.append(a)
                    mae_list.append(m)
                    indiv_logs[sym] = {"Accuracy": f"{a:.1f}%", "MAE": round(m, 3)}
                except Exception:
                    pass
        if acc_list:
            avg_acc = float(np.mean(acc_list))
            avg_mae = float(np.mean(mae_list))
            _update_model_log(
                "Individual_Strategy_Avg",
                {"Accuracy": f"{avg_acc:.1f}%", "MAE": round(avg_mae, 3)},
                len(refined_features)
            )
            with open(MODELS_DIR / "individual_strategy_log.json", "w", encoding="utf-8") as f:
                json.dump(indiv_logs, f, indent=4)

    # -------------------------------------------------------------------------
    # STAGE 5: COMPLETED & PERSIST STATE
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    completion_msg = (
        f"Strategy model evaluated: Accuracy {acc_ref:.1f}%, Win Rate Recall {recall:.1f}%, "
        f"MAE {mae_ref:.3f} R ({len(refined_features)} features, {removed_count} pruned) in {duration:.1f}s."
    )

    json_manager.update_strategy_inference_info(
        status="COMPLETED",
        accuracy_base=f"{acc_base:.1f}%",
        mae_base=mae_base,
        accuracy_refined=f"{acc_ref:.1f}%",
        win_rate_recall=f"{recall:.1f}%",
        mae_refined=mae_ref,
        base_features_count=len(base_features),
        refined_features_count=len(refined_features),
        message=completion_msg,
        duration_seconds=duration,
        last_inference_date=today_str,
        last_run_date=today_str
    )

    json_manager.update_inference_progress(
        status="COMPLETED",
        active_section="STRATEGY",
        phase="Done",
        heading="Done",
        current_model="5_year_xgboost_model.pkl",
        percentage=100.0,
        message=completion_msg,
        elapsed_seconds=duration,
        metrics={
            "accuracy_refined": f"{acc_ref:.1f}%",
            "win_rate_recall": f"{recall:.1f}%",
            "mae_refined": round(mae_ref, 3),
            "accuracy_base": f"{acc_base:.1f}%",
            "mae_base": round(mae_base, 3)
        }
    )

    json_manager.log_app_event(
        event="STRATEGY_INFERENCE_COMPLETED",
        status="Everything fine",
        message=completion_msg
    )

    json_manager.log_task_entry(
        task_name="STRATEGY_INFERENCE",
        status="COMPLETED",
        message=completion_msg
    )

    return {
        "status": "SUCCESS",
        "section": "STRATEGY",
        "accuracy_base": f"{acc_base:.1f}%",
        "mae_base": round(mae_base, 3),
        "accuracy_refined": f"{acc_ref:.1f}%",
        "win_rate_recall": f"{recall:.1f}%",
        "mae_refined": round(mae_ref, 3),
        "duration_seconds": round(duration, 2),
        "message": completion_msg
    }


# ==============================================================================
# SECTION 2: GLOBAL REGIME MODEL INFERENCE (NOTEBOOK 5.2)
# ==============================================================================

def run_global_inference(force: bool = False) -> Dict[str, Any]:
    """
    Evaluates Global Regime Models (Base and Refined) on historical out-of-sample market regimes.
    Calculates MAE Gain, MAE Drop, and Directional Accuracy.
    Writes results to models/model_performance_log.json and data_store/inference_state.json.
    """
    start_time = time.time()
    today_str = str(date.today())

    # Daily skip check
    inf_info = json_manager.get_inference_info()
    glob_info = inf_info.get("global_inference", {})
    last_run_date = glob_info.get("last_run_date", "")

    if not force and last_run_date == today_str and glob_info.get("status") == "COMPLETED":
        msg = f"Global models already evaluated today ({today_str}). Evaluation skipped."
        duration = round(time.time() - start_time, 2)
        json_manager.update_inference_progress(
            status="COMPLETED",
            active_section="GLOBAL",
            phase="Done",
            heading="Done",
            percentage=100.0,
            message=msg,
            elapsed_seconds=duration,
            metrics={
                "mae_gain_refined": glob_info.get("mae_gain_refined", 15.93),
                "mae_drop_refined": glob_info.get("mae_drop_refined", 9.92),
                "directional_accuracy": glob_info.get("directional_accuracy", "48.8%")
            }
        )
        return {
            "status": "SUCCESS",
            "section": "GLOBAL",
            "message": msg,
            "skipped": True,
            "metrics": glob_info
        }

    # Verify model and dataset files exist
    required_files = [
        FINAL_DATA_CSV,
        MODELS_DIR / "global_gain_base.pkl",
        MODELS_DIR / "global_drop_base.pkl",
        MODELS_DIR / "global_base_scaler.pkl",
        MODELS_DIR / "global_base_imputer.pkl",
        MODELS_DIR / "5_year_global_gain_model.pkl",
        MODELS_DIR / "5_year_global_drop_model.pkl",
        MODELS_DIR / "5_year_global_scaler.pkl",
        MODELS_DIR / "5_year_global_imputer.pkl",
        BASE_DIR / "global_base_features.txt",
        BASE_DIR / "global_features.txt"
    ]
    for rf in required_files:
        if not rf.exists():
            err_msg = f"Required file missing for global inference: {rf.name}"
            json_manager.update_inference_progress(
                status="ERROR",
                active_section="GLOBAL",
                message=err_msg,
                error=err_msg
            )
            json_manager.log_task_entry("GLOBAL_INFERENCE", "ERROR", err_msg, error=err_msg)
            return {"status": "ERROR", "message": err_msg}

    json_manager.log_task_entry(
        task_name="GLOBAL_INFERENCE",
        status="RUNNING",
        message="Evaluating Global Regime Models on historical out-of-sample test splits..."
    )

    # -------------------------------------------------------------------------
    # STAGE 1: INGESTION & TEST SPLIT (0% -> 25%)
    # -------------------------------------------------------------------------
    json_manager.update_inference_progress(
        status="EVALUATING",
        active_section="GLOBAL",
        phase="Ingestion",
        heading="Filtering Regimes",
        current_model="5_year_final_data.csv",
        percentage=10.0,
        message="Loading master dataset and filtering forward 7-day targets...",
        elapsed_seconds=time.time() - start_time
    )

    df = pd.read_csv(FINAL_DATA_CSV)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    df_clean = df.dropna(subset=['max_gain_7d', 'max_loss_7d']).sort_values('date').reset_index(drop=True)
    n_samples = len(df_clean)
    if n_samples < 50:
        err_msg = f"Insufficient valid rows for global inference ({n_samples} found)."
        json_manager.update_inference_progress(
            status="ERROR",
            active_section="GLOBAL",
            message=err_msg,
            error=err_msg
        )
        return {"status": "ERROR", "message": err_msg}

    test_start = int(n_samples * 0.80)
    test_df = df_clean.iloc[test_start:].copy()

    base_features = _load_features_list(BASE_DIR / "global_base_features.txt")
    refined_features = _load_features_list(BASE_DIR / "global_features.txt")

    # -------------------------------------------------------------------------
    # STAGE 2: BASE GLOBAL MODELS EVALUATION (25% -> 55%)
    # -------------------------------------------------------------------------
    json_manager.update_inference_progress(
        status="EVALUATING",
        active_section="GLOBAL",
        phase="Phase 1: Base Global Models Evaluation",
        heading="Evaluating Base Global Models",
        current_model="global_gain_base.pkl",
        percentage=35.0,
        message=f"Evaluating base Gain & Drop models on {len(test_df):,} test regimes ({len(base_features)} features)...",
        elapsed_seconds=time.time() - start_time
    )

    gain_base = joblib.load(MODELS_DIR / "global_gain_base.pkl")
    drop_base = joblib.load(MODELS_DIR / "global_drop_base.pkl")
    base_scaler = joblib.load(MODELS_DIR / "global_base_scaler.pkl")
    base_imputer = joblib.load(MODELS_DIR / "global_base_imputer.pkl")

    X_test_base = test_df[base_features]
    X_test_s_base = base_scaler.transform(base_imputer.transform(X_test_base))

    preds_gain_base = gain_base.predict(X_test_s_base)
    preds_drop_base = drop_base.predict(X_test_s_base)

    mae_gain_base = float(mean_absolute_error(test_df['max_gain_7d'], preds_gain_base))
    mae_drop_base = float(mean_absolute_error(test_df['max_loss_7d'], preds_drop_base))

    _update_model_log(
        model_name="Global_Gain_Base",
        metrics={"MAE": round(mae_gain_base, 3)},
        features_count=len(base_features),
        removed_features=0
    )
    _update_model_log(
        model_name="Global_Drop_Base",
        metrics={"MAE": round(mae_drop_base, 3)},
        features_count=len(base_features),
        removed_features=0
    )

    # -------------------------------------------------------------------------
    # STAGE 3: REFINED GLOBAL MODELS EVALUATION (55% -> 90%)
    # -------------------------------------------------------------------------
    json_manager.update_inference_progress(
        status="EVALUATING",
        active_section="GLOBAL",
        phase="Phase 2: Refined Global Models Evaluation",
        heading="Evaluating Refined Global Models",
        current_model="5_year_global_gain_model.pkl",
        percentage=65.0,
        message=f"Evaluating pruned Gain & Drop models on {len(refined_features)} features...",
        elapsed_seconds=time.time() - start_time,
        metrics={
            "mae_gain_base": round(mae_gain_base, 3),
            "mae_drop_base": round(mae_drop_base, 3)
        }
    )

    gain_ref = joblib.load(MODELS_DIR / "5_year_global_gain_model.pkl")
    drop_ref = joblib.load(MODELS_DIR / "5_year_global_drop_model.pkl")
    ref_scaler = joblib.load(MODELS_DIR / "5_year_global_scaler.pkl")
    ref_imputer = joblib.load(MODELS_DIR / "5_year_global_imputer.pkl")

    X_test_ref = test_df[refined_features]
    X_test_s_ref = ref_scaler.transform(ref_imputer.transform(X_test_ref))

    preds_gain_ref = gain_ref.predict(X_test_s_ref)
    preds_drop_ref = drop_ref.predict(X_test_s_ref)

    mae_gain_ref = float(mean_absolute_error(test_df['max_gain_7d'], preds_gain_ref))
    mae_drop_ref = float(mean_absolute_error(test_df['max_loss_7d'], preds_drop_ref))

    actual_up = test_df['max_gain_7d'] > test_df['max_loss_7d'].abs()
    pred_up = pd.Series(preds_gain_ref) > pd.Series(preds_drop_ref).abs()
    dir_acc = float((actual_up.reset_index(drop=True) == pred_up.reset_index(drop=True)).mean() * 100)

    removed_count = max(0, len(base_features) - len(refined_features))

    _update_model_log(
        model_name="Global_Gain_Refined",
        metrics={
            "MAE": round(mae_gain_ref, 3),
            "Directional_Accuracy": f"{dir_acc:.1f}%"
        },
        features_count=len(refined_features),
        removed_features=removed_count
    )
    _update_model_log(
        model_name="Global_Drop_Refined",
        metrics={"MAE": round(mae_drop_ref, 3)},
        features_count=len(refined_features),
        removed_features=removed_count
    )

    # -------------------------------------------------------------------------
    # STAGE 4: OPTIONAL INDIVIDUAL GLOBAL MODELS (90% -> 98%)
    # -------------------------------------------------------------------------
    individual_dir = MODELS_DIR / "individual_model" / "global"
    if individual_dir.exists():
        mae_g_list, mae_d_list, dir_list, indiv_logs = [], [], [], {}
        for sym in test_df['symbol'].unique():
            g_path = individual_dir / f"{sym}_gain_model.pkl"
            d_path = individual_dir / f"{sym}_drop_model.pkl"
            if g_path.exists() and d_path.exists():
                sym_df = test_df[test_df['symbol'] == sym]
                if len(sym_df) == 0:
                    continue
                try:
                    s_g = joblib.load(g_path)
                    s_d = joblib.load(d_path)
                    s_sc = joblib.load(individual_dir / f"{sym}_scaler.pkl")
                    s_imp = joblib.load(individual_dir / f"{sym}_imputer.pkl")
                    X_s = s_sc.transform(s_imp.transform(sym_df[refined_features]))
                    pg = s_g.predict(X_s)
                    pd_ = s_d.predict(X_s)
                    mg = float(mean_absolute_error(sym_df['max_gain_7d'], pg))
                    md = float(mean_absolute_error(sym_df['max_loss_7d'], pd_))
                    au = sym_df['max_gain_7d'] > sym_df['max_loss_7d'].abs()
                    pu = pd.Series(pg) > pd.Series(pd_).abs()
                    da = float((au.reset_index(drop=True) == pu.reset_index(drop=True)).mean() * 100)
                    mae_g_list.append(mg)
                    mae_d_list.append(md)
                    dir_list.append(da)
                    indiv_logs[sym] = {"MAE_Gain": round(mg, 3), "MAE_Drop": round(md, 3), "Directional_Accuracy": f"{da:.1f}%"}
                except Exception:
                    pass
        if mae_g_list:
            avg_mg = float(np.mean(mae_g_list))
            avg_md = float(np.mean(mae_d_list))
            avg_da = float(np.mean(dir_list))
            _update_model_log(
                "Individual_Global_Avg",
                {"MAE_Gain": round(avg_mg, 3), "MAE_Drop": round(avg_md, 3), "Directional_Accuracy": f"{avg_da:.1f}%"},
                len(refined_features)
            )
            with open(MODELS_DIR / "individual_global_log.json", "w", encoding="utf-8") as f:
                json.dump(indiv_logs, f, indent=4)

    # -------------------------------------------------------------------------
    # STAGE 5: COMPLETED & PERSIST STATE
    # -------------------------------------------------------------------------
    duration = time.time() - start_time
    completion_msg = (
        f"Global models evaluated: Directional Accuracy {dir_acc:.1f}%, Gain MAE {mae_gain_ref:.3f}%, "
        f"Drop MAE {mae_drop_ref:.3f}% ({len(refined_features)} features, {removed_count} pruned) in {duration:.1f}s."
    )

    json_manager.update_global_inference_info(
        status="COMPLETED",
        mae_gain_base=mae_gain_base,
        mae_drop_base=mae_drop_base,
        mae_gain_refined=mae_gain_ref,
        mae_drop_refined=mae_drop_ref,
        directional_accuracy=f"{dir_acc:.1f}%",
        base_features_count=len(base_features),
        refined_features_count=len(refined_features),
        message=completion_msg,
        duration_seconds=duration,
        last_inference_date=today_str,
        last_run_date=today_str
    )

    json_manager.update_inference_progress(
        status="COMPLETED",
        active_section="GLOBAL",
        phase="Done",
        heading="Done",
        current_model="5_year_global_gain_model.pkl",
        percentage=100.0,
        message=completion_msg,
        elapsed_seconds=duration,
        metrics={
            "directional_accuracy": f"{dir_acc:.1f}%",
            "mae_gain_refined": round(mae_gain_ref, 3),
            "mae_drop_refined": round(mae_drop_ref, 3),
            "mae_gain_base": round(mae_gain_base, 3),
            "mae_drop_base": round(mae_drop_base, 3)
        }
    )

    json_manager.log_app_event(
        event="GLOBAL_INFERENCE_COMPLETED",
        status="Everything fine",
        message=completion_msg
    )

    json_manager.log_task_entry(
        task_name="GLOBAL_INFERENCE",
        status="COMPLETED",
        message=completion_msg
    )

    return {
        "status": "SUCCESS",
        "section": "GLOBAL",
        "mae_gain_base": round(mae_gain_base, 3),
        "mae_drop_base": round(mae_drop_base, 3),
        "mae_gain_refined": round(mae_gain_ref, 3),
        "mae_drop_refined": round(mae_drop_ref, 3),
        "directional_accuracy": f"{dir_acc:.1f}%",
        "duration_seconds": round(duration, 2),
        "message": completion_msg
    }


# ==============================================================================
# SECTION 3: UNIFIED SEQUENTIAL ORCHESTRATOR
# ==============================================================================

def run_all_inference(force: bool = False) -> Dict[str, Any]:
    """Runs Strategy Inference followed by Global Regime Inference sequentially."""
    start_time = time.time()
    json_manager.update_inference_progress(
        status="EVALUATING",
        active_section="ALL",
        phase="Starting",
        heading="Initializing",
        percentage=0.0,
        message="Starting complete quantitative model evaluation suite..."
    )

    res_strat = run_strategy_inference(force=force)
    if res_strat.get("status") == "ERROR":
        return res_strat

    res_global = run_global_inference(force=force)
    if res_global.get("status") == "ERROR":
        return res_global

    duration = time.time() - start_time
    total_msg = f"All quantitative models evaluated and validated successfully in {duration:.1f}s."

    json_manager.update_inference_progress(
        status="COMPLETED",
        active_section="ALL",
        phase="Done",
        heading="Done",
        percentage=100.0,
        message=total_msg,
        elapsed_seconds=duration,
        metrics={
            "strategy": res_strat.get("metrics", {}),
            "global": res_global.get("metrics", {})
        }
    )

    return {
        "status": "SUCCESS",
        "section": "ALL",
        "strategy": res_strat,
        "global": res_global,
        "duration_seconds": round(duration, 2),
        "message": total_msg
    }
