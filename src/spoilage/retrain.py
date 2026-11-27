"""
Stage 4 Retraining, Evaluation, Model Promotion, and Holdout Prediction Pipeline.

Implements:
- Population Stability Index and Chamber-wise error analysis on later.csv.
- Date-sorted chronological split (70% train / 30% check).
- Automated retraining trigger driven by data drift detection.
- Candidate model training with MLflow tracking (SQLite) and data_sha256 logging.
- Client promotion rule: Promote v2 only if last 30% MAE is at least 8% lower than v1.
- Safe model registry versioning and @champion alias management.
- Leakage-free holdout prediction generation (reports/holdout_predictions.csv).
- Update and completion of reports/findings.yaml.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import yaml

from .data import clean_data, compute_data_sha256
from .drift import run_drift_detection
from .evaluate import load_champion_model
from .train import build_pipeline, train_and_evaluate_model

logger = logging.getLogger(__name__)


def compute_chamber_mae(
    model: Any,
    later_df: pd.DataFrame,
    target_col: str = "spoilage_pct",
    drop_cols: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Compute model MAE on later.csv separately for each value of chamber.

    Args:
        model: Trained scikit-learn or pyfunc model.
        later_df: Cleaned dataframe of later.csv.
        target_col: Name of target column ('spoilage_pct').
        drop_cols: Identifiers to drop before inference.

    Returns:
        Tuple of (mae_by_chamber_dict, worst_served_chamber_name).
    """
    drop_cols = drop_cols or ["record_id", "date"]
    mae_by_chamber: Dict[str, float] = {}

    chambers = sorted(later_df["chamber"].dropna().unique())
    for chamber in chambers:
        chamber_data = later_df[later_df["chamber"] == chamber]
        if len(chamber_data) == 0:
            continue

        X_chamber = chamber_data.drop(columns=[target_col] + [c for c in drop_cols if c in chamber_data.columns])
        y_chamber = chamber_data[target_col]

        preds = model.predict(X_chamber)
        mae = float(mean_absolute_error(y_chamber, preds))
        mae_by_chamber[str(chamber)] = round(mae, 4)

    worst_group = max(mae_by_chamber, key=mae_by_chamber.get) if mae_by_chamber else "None"
    logger.info("Chamber MAE breakdown: %s | Worst served: %s", mae_by_chamber, worst_group)
    return mae_by_chamber, worst_group


def split_later_dataset(
    later_df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort later.csv chronologically by date and split into train (first 70%) and check (last 30%).

    Args:
        later_df: Dataframe containing later.csv data.
        train_ratio: Ratio for the first partition (default: 0.7).

    Returns:
        Tuple of (later_train_df, later_val_df).
    """
    sort_cols = [c for c in ["date", "record_id"] if c in later_df.columns]
    if sort_cols:
        sorted_df = later_df.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        sorted_df = later_df.reset_index(drop=True)

    n_total = len(sorted_df)
    n_train = int(train_ratio * n_total)

    train_part = sorted_df.iloc[:n_train].reset_index(drop=True)
    val_part = sorted_df.iloc[n_train:].reset_index(drop=True)

    logger.info(
        "Sorted later.csv into %d training rows (first %.0f%%) and %d validation rows (last %.0f%%)",
        len(train_part),
        train_ratio * 100,
        len(val_part),
        (1 - train_ratio) * 100,
    )
    return train_part, val_part


def generate_holdout_predictions(
    champion_model: Any,
    holdout_path: str = "data/raw/holdout_features.csv",
    output_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate predictions for holdout_features.csv using the Champion model.
    Guarantees zero target leakage and produces exactly two columns: record_id,prediction.

    Args:
        champion_model: The approved champion model.
        holdout_path: Path to holdout_features.csv.
        output_paths: List of output file destinations.

    Returns:
        DataFrame with columns ['record_id', 'prediction'].
    """
    output_paths = output_paths or [
        "reports/holdout_predictions.csv",
        "predictions/holdout_predictions.csv",
    ]

    p_in = Path(holdout_path)
    if not p_in.is_file():
        raise FileNotFoundError(f"Holdout dataset not found at: {holdout_path}")

    holdout_df = pd.read_csv(p_in)
    if "record_id" not in holdout_df.columns:
        raise KeyError("holdout_features.csv is missing 'record_id' column.")

    record_ids = holdout_df["record_id"].copy()

    # Clean holdout data without dropping record_id until inference prep
    cleaned_holdout = clean_data(
        holdout_df,
        drop_leakage=True,
        drop_identifiers=False,
        is_train=False,
    )

    inference_features = cleaned_holdout.drop(columns=[c for c in ["record_id", "date", "spoilage_pct"] if c in cleaned_holdout.columns])

    raw_preds = champion_model.predict(inference_features)
    preds = np.clip(np.asarray(raw_preds, dtype=float), 0.0, 100.0)

    results_df = pd.DataFrame({
        "record_id": record_ids,
        "prediction": np.round(preds, 4),
    })

    for out_path in output_paths:
        p_out = Path(out_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(p_out, index=False)
        logger.info("Saved holdout predictions (%d rows) to %s", len(results_df), p_out)

    return results_df


def update_findings_yaml(
    findings_path: str,
    psi_dict: Dict[str, float],
    most_drifted_feature: Optional[str],
    mae_by_group: Dict[str, float],
    worst_served_group: str,
    mae_v1_on_check: float,
    mae_v2_on_check: float,
    decision: str,
    stage4_meta: Dict[str, Any],
) -> None:
    """
    Update reports/findings.yaml with Stage 4 monitoring metrics and append stage4 summary.
    Preserves all existing Stage 1 (data_issues) and Stage 2 (tracking) entries.
    """
    p = Path(findings_path)
    if not p.is_file():
        logger.warning("findings.yaml not found at %s; skipping update", findings_path)
        return

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Ensure required top-level identifiers exist
    if "group_id" not in data:
        data["group_id"] = "G056"
    if "problem_id" not in data:
        data["problem_id"] = "MP018"

    # Update monitoring block
    if "monitoring" not in data or not isinstance(data["monitoring"], dict):
        data["monitoring"] = {}

    data["monitoring"]["psi"] = {
        "storage_days": round(float(psi_dict.get("storage_days", 0.0)), 4),
        "temp_set_c": round(float(psi_dict.get("temp_set_c", 0.0)), 4),
        "humidity_pct": round(float(psi_dict.get("humidity_pct", 0.0)), 4),
        "power_cuts_hours": round(float(psi_dict.get("power_cuts_hours", 0.0)), 4),
        "bag_weight_kg": round(float(psi_dict.get("bag_weight_kg", 0.0)), 4),
        "door_openings_per_day": round(float(psi_dict.get("door_openings_per_day", 0.0)), 4),
    }
    data["monitoring"]["most_drifted_feature"] = most_drifted_feature
    data["monitoring"]["mae_by_group"] = {
        "Chamber 1": round(float(mae_by_group.get("Chamber 1", 0.0)), 4),
        "Chamber 2": round(float(mae_by_group.get("Chamber 2", 0.0)), 4),
        "Chamber 3": round(float(mae_by_group.get("Chamber 3", 0.0)), 4),
    }
    data["monitoring"]["worst_served_group"] = worst_served_group
    data["monitoring"]["mae_v1_on_check"] = round(float(mae_v1_on_check), 4)
    data["monitoring"]["mae_v2_on_check"] = round(float(mae_v2_on_check), 4)
    data["monitoring"]["decision"] = decision

    # Append stage4 detailed metadata block
    data["stage4"] = {
        "drift_detected": stage4_meta.get("drift_detected", True),
        "drift_percentage": stage4_meta.get("drift_percentage", 0.0),
        "most_drifted_feature": most_drifted_feature,
        "retraining_triggered": stage4_meta.get("retraining_triggered", True),
        "retraining_status": stage4_meta.get("retraining_status", "completed"),
        "decision": decision,
        "improvement_pct": stage4_meta.get("improvement_pct", 0.0),
        "new_run_id": stage4_meta.get("new_run_id"),
        "registered_version": stage4_meta.get("registered_version"),
        "champion_version": stage4_meta.get("champion_version"),
        "holdout_prediction_file": stage4_meta.get("holdout_prediction_file", "reports/holdout_predictions.csv"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    logger.info("Successfully updated %s with Stage 4 monitoring and summary", p)


def run_stage4_pipeline(params_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Execute the full Stage 4 MLOps pipeline:
    1. Data Drift Detection on train.csv vs later.csv (PSI, KS, Chi-Square).
    2. Chamber-wise error breakdown on later.csv for v1.
    3. Retraining Decision: if drift detected, train candidate v2 on train + first 70% of later.
    4. Validation on last 30% of later.csv.
    5. Evaluation of Client Promotion Rule: promote if MAE is >= 8% lower than v1.
    6. MLflow Model Registry update (@champion alias) if promoted.
    7. Generate final predictions on holdout_features.csv using current Champion.
    8. Update reports/findings.yaml with all monitoring metrics and Stage 4 records.
    """
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # 1. Run Data Drift Detection
    logger.info("Step 1: Running Data Drift Detection...")
    drift_results = run_drift_detection(params_path=params_path)
    overall_drift_detected = drift_results["overall_drift_detected"]
    psi_dict = drift_results["psi"]
    most_drifted_feature = drift_results["most_drifted_feature"]
    drift_percentage = drift_results["drift_percentage"]

    # Load raw datasets
    train_path = params["data"]["raw_train_path"]
    later_path = params["data"]["raw_later_path"]
    holdout_path = params["data"]["raw_holdout_path"]
    target_col = params["data"]["target_col"]
    numeric_cols = params["data"]["numeric_cols"]
    categorical_cols = params["data"]["categorical_cols"]
    drop_cols = params["data"].get("drop_cols", ["record_id", "date"])

    train_raw = pd.read_csv(train_path)
    later_raw = pd.read_csv(later_path)

    # Clean datasets
    train_clean = clean_data(train_raw, drop_leakage=True, drop_identifiers=False, is_train=True)
    later_clean = clean_data(later_raw, drop_leakage=True, drop_identifiers=False, is_train=False)

    # 2. Load existing Champion model (v1)
    logger.info("Step 2: Loading current Champion model...")
    champion_model_v1 = joblib.load("models/model.joblib")

    # Chamber-wise MAE on later.csv
    mae_by_chamber, worst_served_group = compute_chamber_mae(
        model=champion_model_v1,
        later_df=later_clean,
        target_col=target_col,
        drop_cols=drop_cols,
    )

    # 3. Retraining Decision
    retraining_cfg = params.get("retraining", {})
    later_train_ratio = retraining_cfg.get("later_train_ratio", 0.7)
    promotion_threshold = retraining_cfg.get("promotion_threshold", 0.08)
    model_type = retraining_cfg.get("model_type", "gradient_boosting")
    hyperparams = params["models"][model_type]

    # Split later.csv into first 70% and last 30%
    later_train, later_check = split_later_dataset(later_clean, train_ratio=later_train_ratio)

    X_check = later_check.drop(columns=[target_col] + [c for c in drop_cols if c in later_check.columns])
    y_check = later_check[target_col]

    # Compute v1 MAE on last 30%
    mae_v1_on_check = float(mean_absolute_error(y_check, champion_model_v1.predict(X_check)))
    logger.info("Champion v1 MAE on last 30%% of later.csv: %.4f", mae_v1_on_check)

    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    registered_model_name = params["mlflow"]["registered_model_name"]
    champion_alias = params["mlflow"].get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Get current champion version from registry
    current_champion_mv = client.get_model_version_by_alias(registered_model_name, champion_alias)
    current_champion_version = int(current_champion_mv.version)

    candidate_run_id = None
    candidate_version = None
    mae_v2_on_check = mae_v1_on_check
    decision = "keep"
    improvement_pct = 0.0
    active_champion_model = champion_model_v1
    active_champion_version = current_champion_version

    if overall_drift_detected:
        logger.info("Step 3: Significant drift detected. Initiating retraining for candidate v2...")
        # Prepare combined training dataset
        combined_train = pd.concat([train_clean, later_train], ignore_index=True)
        X_train_v2 = combined_train.drop(columns=[target_col] + [c for c in drop_cols if c in combined_train.columns])
        y_train_v2 = combined_train[target_col]

        data_sha256 = compute_data_sha256(train_path)

        candidate_pipe = build_pipeline(
            model_type=model_type,
            hyperparams=hyperparams,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        with mlflow.start_run(run_name=f"{model_type}_v2_retrain") as run:
            candidate_run_id = run.info.run_id

            # Fit candidate model
            candidate_pipe.fit(X_train_v2, y_train_v2)
            train_preds = candidate_pipe.predict(X_train_v2)
            train_mae = float(mean_absolute_error(y_train_v2, train_preds))

            # Evaluate candidate on last 30% check partition
            mae_v2_on_check = float(mean_absolute_error(y_check, candidate_pipe.predict(X_check)))

            # Log parameters and metrics
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("retrained", True)
            mlflow.log_param("combined_train_samples", len(X_train_v2))
            mlflow.log_param("check_samples", len(X_check))
            for k, v in hyperparams.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", mae_v2_on_check)
            mlflow.log_metric("mae_later_check", mae_v2_on_check)
            mlflow.log_metric("mae_v1_check", mae_v1_on_check)

            # Log model artifact with cloudpickle
            mlflow.sklearn.log_model(
                sk_model=candidate_pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

        logger.info(
            "Candidate v2 trained (Run ID: %s). MAE on last 30%%: %.4f (v1: %.4f)",
            candidate_run_id,
            mae_v2_on_check,
            mae_v1_on_check,
        )

        # 4. Evaluate Client Promotion Rule
        # "promote v2 only if its MAE on that last 30% is at least 8% lower than v1's."
        improvement_pct = round(float((mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check), 4)
        logger.info("Validation improvement: %.2f%% (threshold: %.2f%%)", improvement_pct * 100, promotion_threshold * 100)

        if improvement_pct >= promotion_threshold:
            decision = "promote"
            logger.info("Promotion criteria met. Registering v2 and updating champion alias...")

            # Register model version
            model_uri = f"runs:/{candidate_run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            candidate_version = int(mv.version)

            # Update @champion alias
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=champion_alias,
                version=str(candidate_version),
            )
            active_champion_version = candidate_version
            active_champion_model = candidate_pipe

            # Save champion to models/model.joblib
            joblib.dump(candidate_pipe, "models/model.joblib")
            logger.info(
                "Updated Champion alias to version %d and saved models/model.joblib",
                candidate_version,
            )
        else:
            decision = "keep"
            logger.info("Candidate did not exceed 8%% improvement threshold. Retaining Champion v%d.", current_champion_version)
    else:
        logger.info("No significant drift detected. Reusing existing Champion v%d.", current_champion_version)

    # 5. Generate Holdout Predictions using Champion Model
    logger.info("Step 5: Generating predictions on %s using Champion model (v%d)...", holdout_path, active_champion_version)
    pred_cfg = params.get("predictions", {})
    output_paths = [
        pred_cfg.get("holdout_output_path", "reports/holdout_predictions.csv"),
        pred_cfg.get("alt_holdout_output_path", "predictions/holdout_predictions.csv"),
    ]
    holdout_preds_df = generate_holdout_predictions(
        champion_model=active_champion_model,
        holdout_path=holdout_path,
        output_paths=output_paths,
    )

    # 6. Update findings.yaml
    logger.info("Step 6: Updating reports/findings.yaml...")
    stage4_meta = {
        "drift_detected": overall_drift_detected,
        "drift_percentage": drift_percentage,
        "most_drifted_feature": most_drifted_feature,
        "retraining_triggered": overall_drift_detected,
        "retraining_status": "promoted_as_champion" if decision == "promote" else "champion_retained",
        "decision": decision,
        "improvement_pct": round(improvement_pct * 100, 2),
        "new_run_id": candidate_run_id,
        "registered_version": candidate_version,
        "champion_version": active_champion_version,
        "holdout_prediction_file": "reports/holdout_predictions.csv",
    }

    update_findings_yaml(
        findings_path="reports/findings.yaml",
        psi_dict=psi_dict,
        most_drifted_feature=most_drifted_feature,
        mae_by_group=mae_by_chamber,
        worst_served_group=worst_served_group,
        mae_v1_on_check=mae_v1_on_check,
        mae_v2_on_check=mae_v2_on_check,
        decision=decision,
        stage4_meta=stage4_meta,
    )

    return {
        "drift_results": drift_results,
        "mae_by_chamber": mae_by_chamber,
        "worst_served_group": worst_served_group,
        "mae_v1_on_check": mae_v1_on_check,
        "mae_v2_on_check": mae_v2_on_check,
        "improvement_pct": improvement_pct,
        "decision": decision,
        "candidate_run_id": candidate_run_id,
        "candidate_version": candidate_version,
        "champion_version": active_champion_version,
        "holdout_samples": len(holdout_preds_df),
    }
