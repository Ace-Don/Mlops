# Enterprise Financial Fraud Detection MLOps

An end-to-end, hyper-scale Machine Learning Operations (MLOps) project designed to detect financial fraud in real-time. This project evolves a monolithic Jupyter Notebook (`BankSim` dataset) into a distributed, robust, and automated production architecture.

> [!IMPORTANT]
> **New to the project?** Start by reading the [Master Project Architecture Guide & Rationale](file:///d:/Mlops/Financial%20Fraud%20Detection/PROJECT_RATIONALE.md) for a beginner-friendly, step-by-step technical explanation of the entire system architecture, technology choices, and design patterns.

## 🌟 Executive Summary

This repository implements a modern AI serving architecture. It doesn't just train a model—it deploys an **Enterprise Inference API** capable of enriching real-time transactions with an **Online Feature Store** (Redis), tracking telemetry with **Prometheus**, offloading retraining pipelines to **Celery**, and securely testing candidate models on live traffic using **Shadow Deployments**. 

A complete graphical UI is provided via **Streamlit** to manage the entire lifecycle.

---

## 🏗️ Architecture & Technology Stack

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | ZenML | Manages pipeline steps (data loading, preprocessing, evaluating). |
| **Experiment Tracking** | MLflow | Tracks model hyperparameters, performance metrics, and artifacts. |
| **Inference Engine** | FastAPI | High-performance API serving real-time predictions. |
| **Online Feature Store** | Redis | Caches rolling transaction counts for sub-millisecond feature enrichment. |
| **Background Workers** | Celery | Decouples heavy retraining pipelines from the web API thread. |
| **Telemetry & Alerts** | Prometheus & Grafana | Monitors API latency, throughput, and system health in real-time. |
| **Inference Logs DB** | MySQL | Asynchronously logs every prediction and stores ground-truth feedback. |
| **Control Panel** | Streamlit | A unified UI for operators to trigger retrains and manage models. |
| **Containerization** | Docker Compose | Bundles infrastructure (DB, Redis, Prom, Grafana) for easy deployment. |

---

## 🚀 Core Features

### 1. Online Feature Store (Redis Enrichment)
The Machine Learning models require historical aggregated features (e.g., `count_1_day`, `count_7_days`). 
Instead of forcing the client app to calculate these, or running slow SQL queries, the **FastAPI Inference Engine** performs a blazing fast lookup against a **Redis Cache** (`customer_id` -> counts). The counts are dynamically incremented in the background after every prediction.

### 2. Zero-Downtime Shadow Deployments (A/B Serving)
To test new models safely on live traffic without impacting customers:
1. When a retrained model beats Production accuracy, it is promoted to **`STAGING`**.
2. The `POST /reload-model?load_as_shadow=true` endpoint pulls this staging candidate into memory alongside the Active model.
3. Live transactions are evaluated silently by the Shadow Model and logged to the database (`shadow_prediction`), while the user receives the proven Active Model's response.
4. Once verified, `POST /promote-shadow` atomically hot-swaps the pointers in `< 1ms`.

### 3. Decoupled Asynchronous Retraining
Training loops take time. Rather than freezing the API, `POST /retrain` places a JSON message onto a Redis broker. A dedicated **Celery Worker** picks up the job and runs the ZenML training pipeline in the background.

### 4. Closed-Loop Feedback
Upstream audit teams can send verified labels (Legitimate vs. Fraud) to the `POST /feedback` endpoint. This updates the MySQL `inference_logs` table, providing a continuous supply of labeled data for the next training cycle.

---

## 💻 The Streamlit Control Panel

To prevent engineers from manually executing CLI commands or curl requests, this project includes a comprehensive UI.

Run the dashboard:
```bash
streamlit run streamlit_app.py
```

**Features Included:**
* **Real-Time Predictions**: Manually submit transactions and get colored risk scores.
* **Feedback Submissions**: Update historical logs with ground truth labels.
* **Model Operations**: One-click buttons to load Shadow models, Promote to Production, and Trigger Celery retrains.
* **Redis Feature Explorer**: Directly query the online feature store to inspect customer rolling transaction counts cached in RAM.

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have Docker Desktop installed, along with Python 3.9+.

```bash
pip install -r requirements_api.txt
pip install streamlit
```

### 2. Start the Infrastructure
Spin up MySQL, Redis, Prometheus, and Grafana:
```bash
docker-compose up -d
```

### 3. Start the Background Celery Worker
Open a new terminal to process retraining jobs:
```bash
celery -A celery_worker.app worker --loglevel=info -P solo
```

### 4. Start the Enterprise Inference API
Open a new terminal to launch the FastAPI server:
```bash
uvicorn inference_api:app --reload --port 8000
```
*Visit `http://localhost:8000/docs` to view the interactive Swagger documentation.*

### 5. Launch the Streamlit Operations UI
```bash
streamlit run streamlit_app.py
```

---

## 📚 Advanced Documentation References
For more granular details on specific endpoints, database schemas, and metrics, please refer to the dedicated documents generated in this repository:
1. [PROJECT_RATIONALE.md](./PROJECT_RATIONALE.md): Master Project Architecture & Systems Design Guide (End-to-End Walkthrough).
2. [INFERENCE_API_DOCUMENTATION.md](./INFERENCE_API_DOCUMENTATION.md): Deep dive into API routing, request schemas, and connection pooling.
3. [INFRASTRUCTURE_EXPLAINER.md](./INFRASTRUCTURE_EXPLAINER.md): Detailed breakdown of Redis, Celery, Prometheus, and the Shadow Deployment lifecycle logic.
