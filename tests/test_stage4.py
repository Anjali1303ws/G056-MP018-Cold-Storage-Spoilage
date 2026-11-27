"""
Stage 4 Tests: Drift Detection, Retraining Decision Logic,
Holdout Predictions, and findings.yaml Updates.

All tests operate on in-memory synthetic data to ensure 100% deterministic,
isolated, and reliable execution in CI (GitHub Actions) without requiring raw CSV files.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
import yaml

from src.spoilage.drift import (
    calculate_categorical_drift,
    calculate_ks_test,
    calculate_psi,
    detect_dataset_drift,
)
from src.spoilage.retrain import (
    compute_chamber_mae,
    generate_holdout_predictions,
    split_later_dataset,
    update_findings_yaml,
)


@pytest.fixture
def synthetic_drift_datasets():
    """
    Creates reference and actual datasets with controlled drift:
    - storage_days: stable (no drift)
    - power_cuts_hours: strongly shifted (drift detected)
    - crop: identical categorical distribution
    - chamber: skewed distribution
    """
    np.random.seed(42)
    n = 200

    ref_df = pd.DataFrame({
        "storage_days": np.random.normal(loc=40.0, scale=10.0, size=n),
        "temp_set_c": np.random.normal(loc=3.0, scale=0.5, size=n),
        "humidity_pct": np.random.normal(loc=85.0, scale=3.0, size=n),
        "power_cuts_hours": np.random.exponential(scale=5.0, size=n),
        "bag_weight_kg": np.random.normal(loc=50.0, scale=2.0, size=n),
        "door_openings_per_day": np.random.randint(10, 20, size=n).astype(float),
        "crop": np.random.choice(["apple", "potato", "onion"], size=n),
        "chamber": np.random.choice(["Chamber 1", "Chamber 2", "Chamber 3"], size=n),
    })

    act_df = pd.DataFrame({
        # Stable
        "storage_days": np.random.normal(loc=40.0, scale=10.0, size=n),
        "temp_set_c": np.random.normal(loc=3.0, scale=0.5, size=n),
        "humidity_pct": np.random.normal(loc=85.0, scale=3.0, size=n),
        # Severe drift (loc shifted by 30)
        "power_cuts_hours": np.random.exponential(scale=35.0, size=n),
        "bag_weight_kg": np.random.normal(loc=50.0, scale=2.0, size=n),
        "door_openings_per_day": np.random.randint(10, 20, size=n).astype(float),
        "crop": np.random.choice(["apple", "potato", "onion"], size=n),
        "chamber": np.random.choice(["Chamber 1", "Chamber 2", "Chamber 3"], size=n),
    })

    return ref_df, act_df


def test_psi_identical_vs_shifted_distribution():
    """
    Test that calculate_psi returns near-zero for identical distributions
    and exceeds threshold (>0.25) when distribution shifts significantly.
    """
    np.random.seed(42)
    base = pd.Series(np.random.normal(10, 2, 500))
    identical = pd.Series(np.random.normal(10, 2, 500))
    shifted = pd.Series(np.random.normal(25, 2, 500))

    psi_same = calculate_psi(base, identical, num_bins=10)
    psi_diff = calculate_psi(base, shifted, num_bins=10)

    assert psi_same < 0.10, f"Expected low PSI for identical distributions, got {psi_same}"
    assert psi_diff >= 0.25, f"Expected high PSI for shifted distributions, got {psi_diff}"


def test_ks_and_chisquare_tests():
    """
    Test Kolmogorov-Smirnov and Chi-Square statistical drift test helpers.
    """
    np.random.seed(42)
    s1 = pd.Series(np.random.normal(0, 1, 200))
    s2 = pd.Series(np.random.normal(5, 1, 200))
    ks_res = calculate_ks_test(s1, s2, alpha=0.05)
    assert ks_res["drifted"] is True
    assert ks_res["p_value"] < 0.05

    cat1 = pd.Series(["A"] * 100 + ["B"] * 100)
    cat2 = pd.Series(["A"] * 10 + ["B"] * 190)
    cat_res = calculate_categorical_drift(cat1, cat2, alpha=0.05)
    assert cat_res["drifted"] is True
    assert cat_res["p_value"] < 0.05


def test_detect_dataset_drift(synthetic_drift_datasets):
    """
    Test full dataset drift analysis, summary metrics, and overall_drift_detected flag.
    """
    ref_df, act_df = synthetic_drift_datasets
    numeric_cols = [
        "storage_days", "temp_set_c", "humidity_pct",
        "power_cuts_hours", "bag_weight_kg", "door_openings_per_day"
    ]
    categorical_cols = ["crop", "chamber"]

    results = detect_dataset_drift(
        reference_df=ref_df,
        actual_df=act_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        psi_threshold=0.25,
        num_bins=10,
    )

    assert results["overall_drift_detected"] is True
    assert results["most_drifted_feature"] == "power_cuts_hours"
    assert "power_cuts_hours" in results["drifted_features"]
    assert results["total_features_evaluated"] == 8
    assert results["drift_percentage"] > 0.0
    assert isinstance(results["drift_summary"], str)
    assert len(results["drift_summary"]) > 20


def test_split_later_dataset():
    """
    Test chronological sorting and 70/30 partitioning of the monitoring dataset.
    """
    dates = [f"2026-07-{d:02d}" for d in range(1, 31)] + [f"2026-08-{d:02d}" for d in range(1, 31)]
    records = [f"REC{i:03d}" for i in range(len(dates))]

    df = pd.DataFrame({
        "date": dates,
        "record_id": records,
        "val": range(len(dates)),
    })
    # Shuffle order before splitting to verify chronological sort
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_part, val_part = split_later_dataset(shuffled_df, train_ratio=0.7)

    assert len(train_part) + len(val_part) == 60
    assert len(train_part) == 42
    assert len(val_part) == 18

    # Verify that train max date is <= val min date
    assert train_part["date"].max() <= val_part["date"].min()


def test_compute_chamber_mae():
    """
    Test chamber-wise MAE calculation and identifying the worst-served chamber.
    """
    class FixedPredictor:
        def predict(self, X):
            return np.full(len(X), 5.0)

    model = FixedPredictor()
    df = pd.DataFrame({
        "chamber": ["Chamber 1", "Chamber 1", "Chamber 2", "Chamber 2", "Chamber 3"],
        "spoilage_pct": [5.5, 4.5, 9.0, 10.0, 5.0],
        "feature_1": [1, 2, 3, 4, 5],
    })

    mae_dict, worst = compute_chamber_mae(model, df, target_col="spoilage_pct", drop_cols=[])
    assert "Chamber 1" in mae_dict
    assert "Chamber 2" in mae_dict
    assert "Chamber 3" in mae_dict
    assert worst == "Chamber 2"
    assert mae_dict["Chamber 2"] > mae_dict["Chamber 1"]


def test_holdout_predictions_generation():
    """
    Test holdout prediction generation, schema validation, and zero target leakage.
    """
    class MockModel:
        def predict(self, X):
            return np.linspace(5.0, 15.0, len(X))

    model = MockModel()

    with tempfile.TemporaryDirectory() as tmp_dir:
        holdout_csv = Path(tmp_dir) / "holdout_features.csv"
        out_csv = Path(tmp_dir) / "holdout_predictions.csv"

        df_in = pd.DataFrame({
            "record_id": [f"REC_{i}" for i in range(10)],
            "storage_days": [20] * 10,
            "crop": ["apple"] * 10,
            "chamber": ["Chamber 1"] * 10,
        })
        df_in.to_csv(holdout_csv, index=False)

        preds_df = generate_holdout_predictions(
            champion_model=model,
            holdout_path=str(holdout_csv),
            output_paths=[str(out_csv)],
        )

        assert out_csv.is_file()
        loaded = pd.read_csv(out_csv)
        assert list(loaded.columns) == ["record_id", "prediction"]
        assert len(loaded) == 10
        assert not loaded["prediction"].isnull().any()
        assert loaded["record_id"].iloc[0] == "REC_0"


def test_findings_yaml_update_and_schema():
    """
    Test updating findings.yaml preserves Stage 1-2 sections and adds Stage 4 schema.
    """
    initial_findings = {
        "group_id": "G056",
        "problem_id": "MP018",
        "data_issues": {"duplicate_rows": {"present": True, "count": 28}},
        "tracking": {"experiment_name": "cold-storage-spoilage", "champion_version": 1},
        "monitoring": {
            "psi": {
                "storage_days": None,
                "temp_set_c": None,
                "humidity_pct": None,
                "power_cuts_hours": None,
                "bag_weight_kg": None,
                "door_openings_per_day": None,
            },
            "most_drifted_feature": None,
            "mae_by_group": {"Chamber 1": None, "Chamber 2": None, "Chamber 3": None},
            "worst_served_group": None,
            "mae_v1_on_check": None,
            "mae_v2_on_check": None,
            "decision": None,
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        findings_path = Path(tmp_dir) / "findings.yaml"
        with open(findings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(initial_findings, f)

        psi_dict = {
            "storage_days": 0.067,
            "temp_set_c": 0.045,
            "humidity_pct": 0.044,
            "power_cuts_hours": 7.319,
            "bag_weight_kg": 0.022,
            "door_openings_per_day": 0.141,
        }
        mae_by_group = {"Chamber 1": 1.61, "Chamber 2": 5.29, "Chamber 3": 1.30}
        stage4_meta = {
            "drift_detected": True,
            "drift_percentage": 25.0,
            "most_drifted_feature": "power_cuts_hours",
            "retraining_triggered": True,
            "retraining_status": "promoted_as_champion",
            "decision": "promote",
            "improvement_pct": 63.29,
            "new_run_id": "mock_run_123",
            "registered_version": 2,
            "champion_version": 2,
            "holdout_prediction_file": "reports/holdout_predictions.csv",
        }

        update_findings_yaml(
            findings_path=str(findings_path),
            psi_dict=psi_dict,
            most_drifted_feature="power_cuts_hours",
            mae_by_group=mae_by_group,
            worst_served_group="Chamber 2",
            mae_v1_on_check=3.225,
            mae_v2_on_check=1.184,
            decision="promote",
            stage4_meta=stage4_meta,
        )

        with open(findings_path, "r", encoding="utf-8") as f:
            updated = yaml.safe_load(f)

        # Verify Stage 1 and 2 keys are preserved
        assert updated["data_issues"]["duplicate_rows"]["count"] == 28
        assert updated["tracking"]["champion_version"] == 1

        # Verify monitoring section is populated
        assert updated["monitoring"]["most_drifted_feature"] == "power_cuts_hours"
        assert updated["monitoring"]["worst_served_group"] == "Chamber 2"
        assert updated["monitoring"]["decision"] == "promote"
        assert updated["monitoring"]["psi"]["power_cuts_hours"] == 7.319

        # Verify stage4 section is present
        assert updated["stage4"]["champion_version"] == 2
        assert updated["stage4"]["decision"] == "promote"
        assert "timestamp" in updated["stage4"]
