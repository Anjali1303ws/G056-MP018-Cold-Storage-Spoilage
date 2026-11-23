# Cold Storage Spoilage Prediction (Lucknow)

**Group ID:** G056  
**Problem ID:** MP018  
**Faculty:** Avantika Saklani | **Batch:** 75  

## Group Members
- S24CSEU2746 - RatneshKumar
- S24CSEU2750 - AnjaliRathi
- S24CSEU2762 - SHREYASGUPTA
- S24CSEU2758 - VishnuPachauri

## Project Overview
This project builds an end-to-end MLOps pipeline for predicting the spoilage percentage (`spoilage_pct`) of cold-stored potato, onion, and apple lots for the Lucknow Cold Chain Cooperative.

## Pipeline Architecture
- **Data Versioning:** DVC
- **Experiment Tracking & Model Registry:** MLflow (SQLite backend: `sqlite:///mlflow.db`)
- **Pipeline & Modeling:** Scikit-learn Pipeline with StandardScaler, OneHotEncoder, and regression algorithms (Random Forest, Gradient Boosting, Extra Trees).
- **Model Registration:** Best model registered with the `@champion` alias and saved to `models/model.joblib`.
- **API & Deployment:** FastAPI service, Docker, Kubernetes manifests, and automated CI/CD testing.
