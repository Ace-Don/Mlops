# Infrastructure Explainer: Redis, Celery, Prometheus & Grafana

This document provides an in-depth breakdown of the supporting infrastructure components in the **Financial Fraud Detection Inference Service**. It explains what each component is, why it was chosen, and how it directly serves the real-time prediction and MLOps lifecycle.

---

## 1. Redis (The Online Feature Store & Celery Message Broker)

### What is Redis?
Redis (Remote Dictionary Server) is an open-source, in-memory, key-value data structure store. Because it holds all data in RAM rather than reading/writing to a physical disk, it delivers sub-millisecond read and write operations.

### What is its Role in this Application?
In this MLOps architecture, Redis plays a dual role:
1. **Online Feature Store (Data Serving):**
   * **The Problem:** The machine learning model requires historical features to make a prediction (specifically: `count_1_day`, `count_7_days`, and `count_30_days`—representing the number of transactions a customer has made in those windows). Running an SQL aggregation query on a relational database like MySQL during a live prediction request is far too slow (taking 50ms to 500ms+), which violates real-time latency SLAs.
   * **The Solution:** We store and increment these rolling counts directly in Redis using customer-specific keys. When a request to `/predict` comes in, the FastAPI server performs a quick, sub-millisecond lookup to fetch these values, constructs the full feature payload, runs the prediction, and then asynchronously increments the counts in Redis for subsequent transactions.
2. **Celery Task Broker:**
   * Redis acts as a queue/message broker. When a client triggers a heavy job (like `/retrain`), FastAPI pushes a JSON message describing the task into a Redis queue. The Celery worker polls this queue and pulls the task to execute it.

---

## 2. Celery (The Asynchronous Task Queue)

### What is Celery?
Celery is a distributed task queue framework for Python. It allows you to offload heavy, CPU-intensive, or long-running operations to background worker processes that run completely independently of your main web application.

### What is its Role in this Application?
* **Decoupled Training Executions:** 
  * Running a full ZenML feature engineering and training pipeline can take minutes to complete. If the FastAPI web server ran this training pipeline inline inside the request-response thread, the connection would timeout, and the server would freeze, preventing other customers from getting transaction predictions.
  * When a user calls `/retrain`, FastAPI uses Celery (`run_training_pipeline_task.delay()`) to offload the training task. Celery puts the job into Redis, and the background `celery_worker.py` process picks it up and runs it. The web server remains 100% responsive to live prediction traffic.

---

## 3. Prometheus (The Metrics Collector)

### What is Prometheus?
Prometheus is a powerful, open-source monitoring and alerting toolkit. It operates on a **pull model**, meaning it periodically scrapes numerical metrics from configured HTTP endpoints (in our case, the `/metrics` endpoint exposed by FastAPI).

### What is its Role in this Application?
* **Real-Time API Health and Telemetry:**
  * It monitors performance metrics automatically via `prometheus-fastapi-instrumentator`.
  * It measures:
    * **Latency:** How many milliseconds the `/predict` endpoint takes to respond.
    * **Throughput (RPS):** The number of prediction requests processed per second.
    * **Error Rates:** The percentage of HTTP 500 or 400 errors returned to users.
    * **System Metrics:** CPU, memory usage, and connection pool size.

---

## 4. Grafana (The Visualization Dashboard)

### What is Grafana?
Grafana is a multi-platform open-source analytics and interactive visualization web application. It connects to data sources like Prometheus or MySQL and displays that data in real-time, customizable charts, graphs, and alert dashboards.

### What is its Role in this Application?
* **The Operational Dashboard:**
  * Instead of looking at raw text logs or Prometheus queries, developers and MLOps engineers use Grafana to view graphs of request rates, latency histograms, and system health.
  * It acts as the visual command center for monitoring model degradation, request spikes, and infrastructure load.

---

# Endpoint Directory & Importance

| Endpoint | Method | Auth Required | Importance & MLOps Lifecycle Role |
| :--- | :--- | :--- | :--- |
| `GET /` | `GET` | No | **Operator Dashboard:** Renders the HTML interface to quickly see which model versions (Production and Shadow) are active, and checks database connectivity. |
| `/metrics` | `GET` | No | **Monitoring Telemetry:** Exposes the Prometheus metrics scraped by the Prometheus server. |
| `POST /predict` | `POST` | No | **Core Serving:** Accepts transactional data, enriches it with Redis online features, returns the fraud prediction to the customer, and schedules the DB logging background task. |
| `POST /feedback` | `POST` | No | **Closed-Loop Feedback:** Updates a log entry with the true outcome (actual fraud vs. legitimate transaction), generating labeled datasets for subsequent training cycles. |
| `POST /reload-model` | `POST` | Yes (`X-API-Key`) | **Hot Swapping & Shadow Setup:** Instructs the API to pull a model from ZenML. Can load a model as an active model (`load_as_shadow=false`) or safe staging shadow model (`load_as_shadow=true`). |
| `POST /promote-shadow` | `POST` | Yes (`X-API-Key`) | **Atomic Promotion:** Instantly swaps the currently loaded shadow model into the active production slot in memory (takes < 1ms, zero downtime). |
| `POST /retrain` | `POST` | Yes (`X-API-Key`) | **Automated Retraining:** Enqueues the training pipeline asynchronously using Celery to train a new model version based on fresh feedback. |

---

# Operational FAQs

### 1. How do I get my API Key?
The API key is configured using the `API_KEY` environment variable. 
* By default, it is defined in the `.env` file at the root of the project:
  ```env
  API_KEY="super-secret-key"
  ```
* When calling sensitive endpoints (`/reload-model`, `/promote-shadow`, `/retrain`), add the header `X-API-Key: super-secret-key` to your request.

### 2. Why don't I see any MySQL logs yet?
If you connect via MySQL Workbench and run `SELECT * FROM inference_logs;` but find it empty, this is because:
1. **No predictions have been requested yet:** The tables are initialized as empty on startup. You must make at least one request to `POST /predict` (using the Swagger UI at `http://localhost:8000/docs` or a client like `curl`) to write logs to the database.
2. **Background Task Threading:** Predictions are written asynchronously using FastAPI `BackgroundTasks`. If there's an error during insertion, it is caught and printed to the terminal console of the FastAPI process without crashing the API response. Check the FastAPI console log to see if any SQL insertion warnings or errors occurred.

### 3. I got a 500 Error: "No model version found... with version identifier staging"
This is expected behavior if you have not successfully trained a new candidate model yet. 
* **The Lifecycle:** In this architecture, a model is only promoted to the **`staging`** stage (making it eligible to be a Shadow Model) if a retraining loop finishes AND the new candidate model achieves a higher test accuracy than the current Production model.
* **The Fix:** You must trigger a retraining job (via `POST /retrain` or the Streamlit UI). If the retrained model beats the production threshold, ZenML will place it in `staging`. Only then can you successfully call `POST /reload-model?load_as_shadow=true`.

### 4. How do I use the Streamlit Control Panel?
A unified operational dashboard is provided via Streamlit (`streamlit_app.py`) to easily manage all endpoints without writing curl requests or using Swagger UI.
* **Launch Command:** `streamlit run streamlit_app.py`
* **Features:**
  * **Live Predictions & Feedback:** Submit transactions and log verified ground truth labels.
  * **Model Operations:** Buttons to trigger background Celery retrains, fetch staging models as shadow deployments, and atomically promote shadows to production.
  * **Redis Features Tab:** Provides a built-in UI to view the live caching of customer rolling transaction counts directly from the Redis container, eliminating the need for a separate Redis GUI.
