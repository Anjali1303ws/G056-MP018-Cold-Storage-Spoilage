# Cold Storage Spoilage Prediction (Lucknow)

**Group ID:** G056  
**Problem ID:** MP018  
**Faculty:** Avantika Saklani | **Batch:** 75  

## Group Members
- **S24CSEU2746** - Ratnesh Kumar
- **S24CSEU2750** - Anjali Rathi
- **S24CSEU2762** - Shreyas Gupta
- **S24CSEU2758** - Vishnu Pachauri

---

## Project Overview
This repository implements a complete, enterprise-grade, end-to-end MLOps pipeline for predicting the spoilage percentage (`spoilage_pct`) of cold-stored potatoes, onions, and apples for the **Lucknow Cold Chain Cooperative** (Bennett University SCSE3040 MLOps Mini Project).

The pipeline covers raw data versioning, exploratory data quality auditing, model experimentation, MLflow experiment tracking, model registry governance, containerized FastAPI deployment, Kubernetes orchestration, CI testing, statistical data drift monitoring, automated retraining with candidate-champion promotion gates, and holdout batch prediction.

---

## End-to-End Pipeline Architecture

### Stage 1: Data Versioning & Quality Auditing
- **Data Versioning:** DVC configured (`.dvc/`) tracking raw datasets (`train.csv.dvc`, `later.csv.dvc`, `holdout_features.csv.dvc`).
- **Data Quality & Leakage Checks (`src/data_checks.py`):**
  - Detected 28 duplicate rows.
  - Standardized crop categories (`apple`, `potato`, `onion`) by resolving variants (`app.`, `pot.`, `oni.`).
  - Identified and eliminated target leakage column `spoilage_found_at_unloading`.
  - Identified stuck sensor period on `door_openings_per_day` (2026-03-04 to 2026-03-27).
  - Documented in `reports/findings.yaml`.

### Stage 2: Preprocessing, Model Training & Registry
- **Scikit-learn Pipeline:** StandardScaler for numerical features and OneHotEncoder for categoricals.
- **Model Comparison:** Evaluated Random Forest, Extra Trees, and Gradient Boosting.
- **Experiment Tracking:** MLflow tracking with SQLite backend (`sqlite:///mlflow.db`), logging training & validation MAE, hyperparameters, and mandatory `data_sha256` hash.
- **Model Registry & Export:** Best model (Gradient Boosting) registered as `cold-storage-spoilage-regressor` with `@champion` alias (v1) and exported to `models/model.joblib`.

### Stage 3: Containerization, API & CI/CD
- **FastAPI Service (`app/main.py`):**
  - `GET /health`: Health probe returning `{"status": "ok"}`.
  - `POST /predict`: Input validation with Pydantic (`Field` constraints: `power_cuts_hours <= 150`, `bag_weight_kg >= 20`).
  - JSON structured logging of predictions to stdout.
- **Containerization (`Dockerfile`):** Python 3.12-slim base image, multi-stage layer caching, tagged `mp018-api:1.0`.
- **Kubernetes (`k8s/`):** Deployment (`mp018-api`, 3 replicas, readiness & liveness probes) and ClusterIP Service (`mp018-svc`, port 80 -> 8000).
- **GitHub Actions (`.github/workflows/ci.yml`):** Automated CI workflow running pytest on push/PR to `main`.

### Stage 4: Drift Detection, Retraining Decision & Holdout Inference
- **Statistical Drift Monitoring (`src/spoilage/drift.py`):**
  - **Numerical Features:** Population Stability Index (PSI) using 10 quantile bins and Kolmogorov-Smirnov (KS) two-sample tests comparing baseline `train.csv` to monitoring data `later.csv`.
  - **Categorical Features:** Chi-Square tests of independence for `crop` and `chamber`.
  - **Findings:** Significant drift detected on `power_cuts_hours` (PSI = 7.3187) and moderate drift on `door_openings_per_day` (PSI = 0.1415). Overall drift triggered (`overall_drift_detected: true`).
  - **Artifact:** Full feature-by-feature report exported to `reports/drift_report.json`.
- **Chamber Error Breakdown:**
  - Evaluated Champion v1 on `later.csv` by chamber:
    - Chamber 1 MAE: 1.6128
    - Chamber 2 MAE: 5.2934
    - Chamber 3 MAE: 1.3022
  - **Worst served group:** `Chamber 2`.
- **Retraining & Promotion Decision (`src/spoilage/retrain.py`):**
  - `later.csv` sorted chronologically and partitioned into first 70% (training) and last 30% (validation check).
  - Candidate model (v2) trained on cleaned `train.csv` + first 70% of `later.csv`.
  - Evaluated v1 vs v2 on the last 30% check partition:
    - v1 Check MAE: 3.2250
    - v2 Check MAE: 1.1838
    - Error reduction: **63.29%** (far exceeding the 8% client requirement).
  - **Decision:** `promote`.
  - Version 2 registered in MLflow Model Registry, assigned `@champion` alias, and saved to `models/model.joblib`.
- **Holdout Predictions:**
  - Unseen features in `data/raw/holdout_features.csv` (250 rows) scored using the promoted `@champion` model.
  - Zero target leakage; exported with format `record_id,prediction` to `reports/holdout_predictions.csv` and `predictions/holdout_predictions.csv`.
- **Findings Form Update:**
  - `reports/findings.yaml` updated with all PSI scores, group MAEs, worst group, check MAEs, decision (`promote`), and appended `stage4` summary block.

---

## Directory Layout

```text
G056-MP018-Cold-Storage-Spoilage/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI testing on push/PR
├── .dvc/                        # DVC configuration
├── app/
│   └── main.py                  # FastAPI prediction & health service with JSON logging
├── data/
│   └── raw/
│       ├── .gitignore
│       ├── train.csv.dvc        # DVC pointer for training dataset
│       ├── later.csv.dvc        # DVC pointer for monitoring dataset
│       └── holdout_features.csv.dvc # DVC pointer for holdout features
├── k8s/
│   ├── deployment.yaml          # Kubernetes Deployment (3 replicas)
│   └── service.yaml             # Kubernetes Service (mp018-svc)
├── models/
│   └── model.joblib             # Active Champion model binary (v2)
├── predictions/
│   └── holdout_predictions.csv  # Holdout batch predictions (record_id, prediction)
├── reports/
│   ├── findings.yaml            # Project findings, tracking, monitoring & Stage 4
│   ├── drift_report.json        # Statistical feature-level drift report
│   └── holdout_predictions.csv  # Holdout batch predictions
├── src/
│   ├── data_checks.py           # Stage 1 data quality checks
│   ├── train.py                 # Stage 2 training & experiment tracking entrypoint
│   ├── retrain.py               # Stage 4 drift, retraining & holdout entrypoint
│   └── spoilage/
│       ├── __init__.py          # Package interface
│       ├── data.py              # Data cleaning and normalization
│       ├── drift.py             # PSI, KS, Chi-Square drift detection
│       ├── evaluate.py          # Model loading and inference utilities
│       ├── retrain.py           # Retraining pipeline, chamber MAE, promotion logic
│       └── train.py             # Pipeline builder and model trainer
├── tests/
│   ├── conftest.py              # Pytest configuration and path resolution
│   ├── test_model.py            # Stage 2 & 3 unit tests
│   └── test_stage4.py           # Stage 4 drift, retraining, holdout & findings tests
├── Dockerfile                   # Production Docker container definition
├── params.yaml                  # Unified project configuration
├── pytest.ini                   # Pytest test discovery settings
├── requirements.txt             # Pinned dependencies
├── GROUP.md                     # Group & member information
├── PROBLEM.md                   # Problem statement & business requirements
├── STUDENT_HANDBOOK.md          # Assignment rubrics & instructions
└── README.md                    # Project documentation
```

---

## How to Run

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Stage 2 Training Pipeline
Trains baseline models, logs experiments to MLflow SQLite backend, and registers initial champion v1:
```bash
python src/train.py
```

### 3. Run Stage 4 Drift Detection & Automated Retraining
Performs drift detection (PSI, KS, Chi-Square), evaluates chamber error, triggers retraining, compares candidate vs champion, updates registry alias `@champion`, scores holdout set, and updates `reports/findings.yaml`:
```bash
python src/retrain.py
```

### 4. Run Pytest Suite
Runs all 11 isolated unit and integration tests (zero external network or raw CSV dependency):
```bash
pytest -v
```

### 5. Start the FastAPI Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- **Health Check:** `curl http://localhost:8000/health`
- **Predict:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"storage_days": 45, "temp_set_c": 3.0, "humidity_pct": 88.0, "power_cuts_hours": 12.0, "bag_weight_kg": 50.0, "door_openings_per_day": 15, "crop": "potato", "chamber": "Chamber 2"}'
```

### 6. Container Execution
```bash
docker build -t mp018-api:1.0 .
docker run -p 8000:8000 mp018-api:1.0
```

---

## Verification & Key Results

| Metric / Check | Value / Status | Description |
|---|---|---|
| **Overall Drift Detected** | `True` | Driven by significant distribution shift in `power_cuts_hours` |
| **Most Drifted Feature** | `power_cuts_hours` | Population Stability Index (PSI) = `7.3187` |
| **Moderate Drift** | `door_openings_per_day` | Population Stability Index (PSI) = `0.1415` |
| **Chamber MAE Breakdown** | `Chamber 1`: 1.6128, `Chamber 2`: 5.2934, `Chamber 3`: 1.3022 | Error calculated on monitoring dataset (`later.csv`) |
| **Worst Served Group** | `Chamber 2` | Highest MAE across chambers |
| **v1 Check MAE** | `3.2250` | Evaluated on last 30% of `later.csv` |
| **v2 Candidate MAE** | `1.1838` | Evaluated on last 30% of `later.csv` |
| **Validation Improvement** | **63.29%** | Exceeds the 8% client threshold required for promotion |
| **Promotion Decision** | `promote` | Candidate promoted to Champion |
| **Active Champion Version** | `2` | Alias `@champion` pointed to Version 2 in MLflow Model Registry |
| **Holdout Predictions** | 250 rows | Generated in `reports/holdout_predictions.csv` |
| **Pytest Suite** | **11 passed** | 100% test pass rate across data checks, tracking, drift, and retraining |
| **CI Status** | **Passing** | GitHub Actions workflow verified |
