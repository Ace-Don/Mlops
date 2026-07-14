# End-to-End (E2E) MLOps Project Architecture & Guide

Welcome to the comprehensive guide for the **End-to-End (E2E) MLOps Template**. This document explains exactly what this project does, how the pipelines work together, how data and metadata flow through the system, and how to run and inspect everything locally on your machine or in Docker.

---

## 1. High-Level Project Overview

This project is a production-grade, modular **Machine Learning Operations (MLOps)** system built to automate the full lifecycle of a Machine Learning model using three core industry standard frameworks:

1. **ZenML (Orchestrator & Lineage Engine)**: Connects all code steps into Directed Acyclic Graphs (DAGs), passes data between steps, caches intermediate results, and records full artifact lineage.
2. **MLflow (Experiment Tracker & Model Registry)**: Logs hyperparameters, tracks evaluation metrics (train/test accuracy), manages model versions, and tags lifecycle stages (`production`, `champion`).
3. **Evidently AI (Data Quality & Drift Validator)**: Monitors incoming batch inference data against the original training baseline to detect data drift before generating predictions.

---

## 2. The Three Core Pipelines

The project is structured around three independent but interconnected pipelines executed via `run.py`.

```mermaid
graph TD
    subgraph Training Pipeline
        A[data_loader] --> B[data_preprocessor]
        B --> C[model_trainer]
        B --> D[model_evaluator]
        C --> D
        D --> E[compute_performance_metrics_on_current_data]
        E --> F[promote_with_metric_compare]
        F --> G[notify_on_success]
    end

    subgraph Deployment Pipeline
        H[deployment_deploy] --> I[notify_on_success]
        F -.-|Promotes Model to 'production'|-> H
    end

    subgraph Batch Inference Pipeline
        J[data_loader] --> K[inference_data_preprocessor]
        K --> L[evidently_report_step]
        L --> M[drift_quality_gate]
        M --> N[inference_predict]
        N --> O[notify_on_success]
        H -.-|Deploys Prediction Service / Model|-> N
    end
```

---

### Pipeline 1: Training Pipeline (`e2e_use_case_training`)
**Goal**: Load raw data, train a classifier, evaluate its accuracy against strict quality gates, and automatically promote it if it beats the existing model.

* **`data_loader`**: Reads the Breast Cancer dataset (features + target column) and exposes parameters (`random_state`) for reproducibility.
* **`data_preprocessor`**: Cleans data (`drop_na`), applies feature scaling/normalization, and splits data into `train` and `test` datasets (`test_size=0.2`).
* **`model_trainer`**: Trains a machine learning model (configured via YAML, such as `RandomForestClassifier` or `SGDClassifier`), logs training parameters to **MLflow**, and registers the model in the **MLflow Model Registry**.
* **`model_evaluator`**: Computes train and test accuracy on the freshly trained model. Logs metrics to **MLflow**. Enforces quality gates (`min_train_accuracy`, `min_test_accuracy`)—if accuracy falls below thresholds, the pipeline halts.
* **`compute_performance_metrics_on_current_data`**: Compares the new model's performance against the currently promoted model on the latest dataset.
* **`promote_with_metric_compare`**: If the newly trained model outperforms the currently active production model, this step updates the **MLflow Model Registry** by assigning the `'production'` (or `'champion'`) alias to the new version.
* **`notify_on_success` / `notify_on_failure`**: Sends webhooks (Discord/Slack/Email) notifying the team of pipeline outcome.

---

### Pipeline 2: Deployment Pipeline (`e2e_use_case_deployment`)
**Goal**: Take the currently promoted `'production'` model from the registry and deploy it as an active prediction service.

* **`deployment_deploy`**: Queries the **MLflow Model Registry** for the version tagged `'production'`, loads its artifact, and deploys a REST API prediction service using the `MLFlowDeploymentService` component.
* **`notify_on_success` / `notify_on_failure`**: Alerts the team when a new model deployment goes live.

---

### Pipeline 3: Batch Inference Pipeline (`e2e_use_case_batch_inference`)
**Goal**: Ingest new unseen data, validate data quality and drift against the training baseline, and generate predictions.

* **`data_loader`**: Ingests new records requiring inference.
* **`inference_data_preprocessor`**: Applies the exact same transformations and normalization rules used during training.
* **`evidently_report_step`**: Uses **Evidently AI** to compute statistical distribution differences between the training dataset and the new inference dataset, generating detailed drift reports.
* **`drift_quality_gate`**: Evaluates drift score against safety thresholds (`max_drift_threshold`). If the data has drifted too severely, inference stops to prevent erroneous predictions.
* **`inference_predict`**: Sends the preprocessed dataset to the deployed MLflow model service to generate `predicted` probabilities/classes. *(Note: On Windows where background daemon services are unsupported, this step intelligently falls back to loading the model artifact directly into memory for local prediction).*
* **`notify_on_success` / `notify_on_failure`**: Confirms batch predictions completion.

---

## 3. Configuration Management (`configs/`)

Instead of hardcoding parameters in Python files, the project uses clean YAML configuration files:
* **`configs/train_config.yaml`**: Sets hyperparameters for `model_trainer`, accuracy thresholds (`min_train_accuracy: 0.8`), and feature engineering flags (`normalize: true`).
* **`configs/deployer_config.yaml`**: Configures prediction service parameters.
* **`configs/inference_config.yaml`**: Sets drift thresholds (`max_drift_threshold: 0.5`) and dataset options for batch inference.

---

## 4. Storage & Metadata Architecture

The system cleanly separates **Relational Pointers & Execution Catalog** from **Heavy Files & Model Payloads**:

```
C:\Users\Nonso\AppData\Roaming\zenml\local_stores\cd2e5fc8-aad1-44db-855d-1d18bbe0cb91\
│
├── 1. ZenML SQLite Database  (default_zen_store.db)
│      ├── Pipeline definitions & Run IDs
│      ├── Step execution states (Completed/Failed) & timing
│      └── Lineage index connecting steps to input/output files
│
├── 2. ZenML Artifact Files   (artifacts\)
│      ├── Serialized Pandas DataFrames (.parquet / binary)
│      ├── Scikit-Learn Classifier objects (.pkl)
│      └── Evidently HTML/JSON Data Drift Reports
│
└── 3. MLflow Tracking & Registry (mlruns\)
       ├── Experiment Tracking: Hyperparameters, Accuracy metrics, & Run timestamps
       ├── Model Registry: Lifecycle aliases ('production', 'champion') & versioning
       └── Packaged MLflow Models (MLmodel, conda.yaml, requirements.txt)
```

---

## 5. How to Run & Inspect Your Project

### Running the End-to-End Pipeline Locally
To run all three pipelines (**Training -> Deployment -> Batch Inference**) sequentially:
```powershell
python run.py
```

### Running Specific Options
* **Run ONLY Training**: `python run.py --no-inference --no-deployment`
* **Run ONLY Batch Inference**: `python run.py --only-inference`
* **Run without cache**: `python run.py --no-cache`

### Viewing Experiment & Model Dashboard (MLflow UI)
To open your web dashboard and inspect all accuracy metrics, hyperparameters, and registered model versions:
```powershell
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --backend-store-uri "file:/C:/Users/Nonso/AppData/Roaming/zenml/local_stores/cd2e5fc8-aad1-44db-855d-1d18bbe0cb91/mlruns"
```
Open **http://localhost:5000** in your browser once started.

---

## 6. Summary of Key Files

| File / Folder | Description |
| :--- | :--- |
| **`run.py`** | Main entrypoint command-line script executing the pipelines with Click arguments. |
| **`pipelines/`** | Defines the DAG execution order (`training.py`, `deployment.py`, `batch_inference.py`). |
| **`steps/`** | Modular Python functions decorated with `@step` containing the actual ML logic. |
| **`configs/`** | YAML configuration files controlling hyperparameters, thresholds, and datasets. |
| **`requirements.txt`** | Complete pinned dependencies (`zenml==0.94.2`, `mlflow==3.14.0`, `evidently`, etc.). |
