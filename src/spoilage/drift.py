"""
Data drift detection module for Cold Storage Spoilage Prediction.

Provides statistical methods for monitoring feature drift between reference (train.csv)
and production/monitoring data (later.csv), supporting:
- Population Stability Index (PSI) with 10 quantile bins for numerical features.
- Kolmogorov-Smirnov (KS) two-sample test for numerical distribution shifts.
- Chi-Square test of independence for categorical distributions.
- Feature-wise drift reporting, drift percentage, most drifted feature, and summary.
- Export to JSON report (reports/drift_report.json).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
import yaml

from .data import clean_data

logger = logging.getLogger(__name__)


def calculate_psi(
    reference: pd.Series,
    actual: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate the Population Stability Index (PSI) between reference and actual distributions.

    Uses 10 quantile bins defined on the reference series. Bin edges are bounded
    by [-inf, +inf] to capture out-of-range observations. An epsilon smoothing term
    prevents division by zero or log of zero.

    Interpretation:
    - PSI < 0.10: No significant distribution change.
    - 0.10 <= PSI < 0.25: Moderate distribution shift.
    - PSI >= 0.25: Significant distribution shift (drift detected).

    Args:
        reference: Baseline/training feature Series.
        actual: Incoming/later feature Series.
        num_bins: Number of quantile bins (default: 10).
        epsilon: Small smoothing constant for zero counts.

    Returns:
        float: Computed Population Stability Index.
    """
    ref_clean = reference.dropna()
    act_clean = actual.dropna()

    if len(ref_clean) == 0 or len(act_clean) == 0:
        logger.warning("Empty series provided for PSI calculation; returning 0.0")
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_binned = pd.cut(ref_clean, bins=bin_edges, right=True)
    act_binned = pd.cut(act_clean, bins=bin_edges, right=True)

    ref_counts = ref_binned.value_counts(sort=False)
    act_counts = act_binned.value_counts(sort=False)

    total_ref = len(ref_clean)
    total_act = len(act_clean)
    num_intervals = len(ref_counts)

    ref_pct = (ref_counts + epsilon) / (total_ref + epsilon * num_intervals)
    act_pct = (act_counts + epsilon) / (total_act + epsilon * num_intervals)

    psi_val = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
    return float(psi_val)


def calculate_ks_test(
    reference: pd.Series,
    actual: pd.Series,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform a two-sample Kolmogorov-Smirnov test for numerical features.

    Args:
        reference: Reference feature series.
        actual: Monitoring/actual feature series.
        alpha: Significance level threshold (default: 0.05).

    Returns:
        Dict containing statistic, p_value, and drift boolean flag.
    """
    ref_clean = reference.dropna()
    act_clean = actual.dropna()

    if len(ref_clean) == 0 or len(act_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drifted": False}

    res = stats.ks_2samp(ref_clean, act_clean)
    p_val = float(res.pvalue)
    statistic = float(res.statistic)

    return {
        "statistic": statistic,
        "p_value": p_val,
        "drifted": bool(p_val < alpha),
    }


def calculate_categorical_drift(
    reference: pd.Series,
    actual: pd.Series,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform a Chi-Square test of independence to assess categorical feature drift.

    Args:
        reference: Reference categorical series.
        actual: Incoming categorical series.
        alpha: Significance level threshold (default: 0.05).

    Returns:
        Dict containing chi2 statistic, p_value, and drift boolean flag.
    """
    ref_clean = reference.dropna().astype(str)
    act_clean = actual.dropna().astype(str)

    all_categories = sorted(list(set(ref_clean.unique()).union(set(act_clean.unique()))))
    ref_counts = ref_clean.value_counts().reindex(all_categories, fill_value=0)
    act_counts = act_clean.value_counts().reindex(all_categories, fill_value=0)

    contingency = np.array([ref_counts.values, act_counts.values])

    if contingency.sum() == 0 or len(all_categories) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "drifted": False}

    res = stats.chi2_contingency(contingency)
    p_val = float(res.pvalue)
    statistic = float(res.statistic)

    return {
        "statistic": statistic,
        "p_value": p_val,
        "drifted": bool(p_val < alpha),
    }


def detect_dataset_drift(
    reference_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    psi_threshold: float = 0.25,
    num_bins: int = 10,
    categorical_alpha: float = 0.05,
    drift_share_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Run full data drift analysis comparing reference dataset to actual dataset.

    Args:
        reference_df: Cleaned baseline data (train.csv).
        actual_df: Cleaned monitoring data (later.csv).
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.
        psi_threshold: Threshold above which numeric drift is flagged (default: 0.25).
        num_bins: Number of quantile bins for PSI (default: 10).
        categorical_alpha: p-value significance threshold for categorical tests.
        drift_share_threshold: Ratio of drifted features required to flag overall drift.

    Returns:
        Dict with keys:
            - feature_wise_drift: Dict[str, Dict] detailed metrics for each feature.
            - psi: Dict[str, float] for numeric columns.
            - categorical_drift: Dict[str, Dict] for categorical columns.
            - most_drifted_feature: Name of the feature with highest drift.
            - drifted_features: List of feature names with confirmed drift.
            - total_features_evaluated: Total count of evaluated features.
            - drifted_feature_count: Count of drifted features.
            - drift_percentage: Percentage of evaluated features displaying drift.
            - overall_drift_detected: Boolean flag indicating if retraining is warranted.
            - drift_summary: Human-readable narrative summary.
    """
    feature_report: Dict[str, Dict[str, Any]] = {}
    psi_scores: Dict[str, float] = {}
    categorical_drift: Dict[str, Dict[str, Any]] = {}
    drifted_features: List[str] = []

    # 1. Numerical Features: Calculate PSI and KS test
    for col in numeric_cols:
        if col not in reference_df.columns or col not in actual_df.columns:
            logger.warning("Numeric column '%s' missing from reference or actual dataset.", col)
            continue

        ref_series = reference_df[col]
        act_series = actual_df[col]

        psi_val = calculate_psi(ref_series, act_series, num_bins=num_bins)
        ks_res = calculate_ks_test(ref_series, act_series)
        is_drifted = bool(psi_val >= psi_threshold or ks_res["drifted"])

        psi_scores[col] = psi_val
        feature_report[col] = {
            "type": "numeric",
            "psi": round(psi_val, 6),
            "ks_statistic": round(ks_res["statistic"], 6),
            "ks_p_value": round(ks_res["p_value"], 6),
            "drift_detected": is_drifted,
            "drift_severity": "significant" if psi_val >= 0.25 else ("moderate" if psi_val >= 0.10 else "minimal"),
        }
        if is_drifted:
            drifted_features.append(col)

    # 2. Categorical Features: Calculate Chi-Square test
    for col in categorical_cols:
        if col not in reference_df.columns or col not in actual_df.columns:
            logger.warning("Categorical column '%s' missing from reference or actual dataset.", col)
            continue

        ref_series = reference_df[col]
        act_series = actual_df[col]

        cat_res = calculate_categorical_drift(ref_series, act_series, alpha=categorical_alpha)
        is_drifted = cat_res["drifted"]

        categorical_drift[col] = cat_res
        feature_report[col] = {
            "type": "categorical",
            "chi2_statistic": round(cat_res["statistic"], 6),
            "p_value": round(cat_res["p_value"], 6),
            "drift_detected": is_drifted,
            "drift_severity": "significant" if is_drifted else "minimal",
        }
        if is_drifted:
            drifted_features.append(col)

    # 3. Determine most drifted feature (highest PSI among numeric features)
    most_drifted_feature: Optional[str] = None
    if psi_scores:
        most_drifted_feature = max(psi_scores, key=psi_scores.get)

    total_evaluated = len(numeric_cols) + len(categorical_cols)
    drifted_count = len(drifted_features)
    drift_share = (drifted_count / total_evaluated) if total_evaluated > 0 else 0.0
    drift_percentage = round(drift_share * 100.0, 2)

    # Overall drift: either any feature exceeds severe PSI threshold, or drift share exceeds threshold
    overall_drift_detected = bool(drifted_count > 0 and (
        drift_share >= drift_share_threshold or
        (most_drifted_feature is not None and psi_scores.get(most_drifted_feature, 0.0) >= psi_threshold)
    ))

    summary_text = (
        f"Drift analysis evaluated {total_evaluated} features ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical). "
        f"{drifted_count} features drifted ({drift_percentage}%). "
        f"Most drifted feature: '{most_drifted_feature}' with PSI={psi_scores.get(most_drifted_feature, 0.0):.4f}. "
        f"Overall drift detected: {overall_drift_detected}."
    )

    logger.info(summary_text)

    return {
        "feature_wise_drift": feature_report,
        "psi": psi_scores,
        "categorical_drift": categorical_drift,
        "most_drifted_feature": most_drifted_feature,
        "drifted_features": drifted_features,
        "total_features_evaluated": total_evaluated,
        "drifted_feature_count": drifted_count,
        "drift_percentage": drift_percentage,
        "overall_drift_detected": overall_drift_detected,
        "drift_summary": summary_text,
    }


def save_drift_report(
    report_data: Dict[str, Any],
    output_path: str = "reports/drift_report.json",
) -> Path:
    """
    Save the drift analysis findings as structured JSON.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved drift report to %s", p)
    return p


def run_drift_detection(params_path: str = "params.yaml") -> Dict[str, Any]:
    """
    High-level entrypoint for executing Stage 4 data drift detection from configuration.

    1. Loads paths and columns from params.yaml.
    2. Reads and cleans train.csv and later.csv.
    3. Runs detect_dataset_drift.
    4. Saves reports/drift_report.json.
    5. Returns complete drift results.
    """
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_cfg = params.get("data", {})
    train_path = data_cfg.get("raw_train_path", "data/raw/train.csv")
    later_path = data_cfg.get("raw_later_path", "data/raw/later.csv")
    numeric_cols = data_cfg.get("numeric_cols", [])
    categorical_cols = data_cfg.get("categorical_cols", [])

    drift_cfg = params.get("drift", {})
    psi_threshold = drift_cfg.get("psi_threshold", 0.25)
    num_bins = drift_cfg.get("psi_bins", 10)
    categorical_alpha = drift_cfg.get("categorical_alpha", 0.05)
    drift_share_threshold = drift_cfg.get("drift_share_threshold", 0.15)
    report_path = drift_cfg.get("report_path", "reports/drift_report.json")

    ref_raw = pd.read_csv(train_path)
    act_raw = pd.read_csv(later_path)

    ref_clean = clean_data(
        ref_raw,
        drop_leakage=True,
        drop_identifiers=False,
        is_train=True,
        leakage_cols=data_cfg.get("leakage_cols"),
        drop_cols=data_cfg.get("drop_cols"),
    )
    act_clean = clean_data(
        act_raw,
        drop_leakage=True,
        drop_identifiers=False,
        is_train=False,
        leakage_cols=data_cfg.get("leakage_cols"),
        drop_cols=data_cfg.get("drop_cols"),
    )

    results = detect_dataset_drift(
        reference_df=ref_clean,
        actual_df=act_clean,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        psi_threshold=psi_threshold,
        num_bins=num_bins,
        categorical_alpha=categorical_alpha,
        drift_share_threshold=drift_share_threshold,
    )

    save_drift_report(results, output_path=report_path)

    return results
