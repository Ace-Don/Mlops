# Enterprise Financial Fraud Inference Service (`inference_api.py`)
**Technical Manual, Architectural Rationale, and Integration Guide**

This document provides a comprehensive technical breakdown of the `inference_api.py` file, detailing its features, architectural decisions, endpoints, schemas, and structural linkages within the Financial Fraud Detection MLOps project.

---

## 1. System Architecture & Component Interaction

The `inference_api.py` is a high-performance, real-time web service built with **FastAPI**. It sits at the serving layer of the MLOps pipeline, acting as the interface between transactional clients and the underlying Machine Learning model registry.

The following sequence diagram illustrates the lifecycle of the service, from startup through prediction, event-driven retraining, and model hot-swapping:

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI Server
    participant Redis as Redis Feature Store / Broker
    database DB as MySQL Database
    participant Celery as Celery Worker
    participant ZenML as ZenML Registry

    Note over API, ZenML: Lifespan Startup Process
    API->>DB: Establish Async MySQL Connection Pool
    API->>Redis: Establish Async Redis Connection Pool
    API->>ZenML: Fetch active "PRODUCTION" Model
    ZenML-->>API: Load Model to app.state.models

    Note over Client, API: Real-Time Prediction Request
    Client->>API: POST /predict (customer_id, amount)
    Note over API: 1. Fetch rolling transaction counts from Redis
    Note over API: 2. Preprocess & Predict (Timeout: 2s)
    API-->>Client: Returns JSON Prediction Response
    API->>DB: Log prediction async (Background Task)
    API->>Redis: Increment rolling counts async (Background Task)

    Note over Client, Celery: Triggering Decoupled Retraining
    Client->>API: POST /retrain (X-API-Key Authorized)
    API->>Redis: Enqueue Celery Task (run_training_pipeline_task)
    API-->>Client: Returns "Retraining job queued"
    Celery->>Redis: Dequeue Task
    Note over Celery, ZenML: Celery executes pipelines.py steps,<br/>logs run to MLflow, updates ZenML registry

    Note over Client, API: Shadow Deployments & Hot-Swapping
    Client->>API: POST /reload-model?load_as_shadow=true
    API->>ZenML: Fetch latest artifacts from STAGING stage
    ZenML-->>API: Load new version as Shadow Model
    Note over Client, API: A/B Serving: Shadow evaluates traffic silently
    Client->>API: POST /promote-shadow (Empty Payload)
    Note over API: Atomic hot-swap pointer (1ms)
    API-->>Client: Shadow model is now Active
```

---

## 2. Core Features & Architectural Nuances (The "Why")

### A. Online Feature Store (Redis)
*   **Implementation:** The API connects to a Redis instance via `redis.asyncio`. When a prediction request arrives, it only requires the `customer_id` and raw transaction data. The API queries Redis for `count_1_day`, `count_7_days`, etc.
*   **Rationale:** Clients shouldn't compute aggregations on the fly. An online feature store provides sub-millisecond lookup times for historical aggregations, heavily reducing client-side logic and preventing data drift.

### B. Shadow Deployments (A/B Serving)
*   **Implementation:** `/reload-model?load_as_shadow=true` loads a candidate model into memory alongside the active model. Incoming requests are evaluated by *both* models. The candidate's output is silently logged to the database (`shadow_prediction` column), while the user receives the active model's response.
*   **Rationale:** Safely tests new models on live traffic without risking customer impact. Once validation is complete, `/promote-shadow` instantly swaps the models.

### C. Advanced Observability (Prometheus)
*   **Implementation:** Integrating `prometheus-fastapi-instrumentator` automatically mounts a `/metrics` endpoint.
*   **Rationale:** Provides high-resolution metrics on request latency, throughput, error rates, and traffic volume. These metrics are scraped by Prometheus and visualized in Grafana.

### D. Modern Task Queues (Celery)
*   **Implementation:** The `/retrain` endpoint drops a message onto a Redis broker via Celery (`.delay()`), rather than polling a MySQL table.
*   **Rationale:** Celery provides robust message brokering, retries, concurrency, and worker scaling out-of-the-box, replacing inefficient database polling.

### E. Automated Feedback Loops
*   **Implementation:** A `POST /feedback` endpoint accepts `prediction_id` and `actual_fraud` labels, updating the `inference_logs` table.
*   **Rationale:** Continuous learning requires ground truth. This webhook allows upstream review systems to label past transactions as fraudulent or legitimate, creating a labeled dataset for the next training cycle.

### F. Fail-Fast Lifespan & Connection Pools
*   **Implementation:** Database pools (`aiomysql`) and model loading occur in the FastAPI `lifespan` event.
*   **Rationale:** Prevents orchestrators (Kubernetes) from routing traffic to broken nodes. Connection pools reduce TCP latency overhead from ~50ms to <2ms.

---

## 3. Database Schema Specifications

### Table: `inference_logs`
Logs every transaction evaluated by the API for continuous monitoring and dataset versioning.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `INT` | `PRIMARY KEY AUTO_INCREMENT`. |
| `timestamp` | `DATETIME` | The exact UTC time the prediction request was processed. |
| `input_features` | `TEXT` | JSON string representation of the raw feature inputs. |
| `prediction` | `INT` | Model classification (e.g. `0` = legitimate, `1` = fraud). |
| `confidence` | `DOUBLE` | The probability score assigned to the chosen class. |
| `probability_legit`| `DOUBLE` | Raw output probability for the legitimate class (0). |
| `probability_fraud`| `DOUBLE` | Raw output probability for the fraudulent class (1). |
| `ground_truth` | `INT` | The verified outcome (supplied post-prediction by `/feedback`). |
| `ground_truth_timestamp` | `DATETIME` | When the ground-truth outcome was logged. |
| `model_version` | `VARCHAR(255)`| The ZenML model version name that evaluated the transaction. |
| `shadow_prediction` | `INT` | The prediction made silently by the shadow candidate model. |
| `api_response_time_ms` | `DOUBLE` | Latency of the prediction process in milliseconds. |

---

## 4. API Endpoint Directory

Sensitive operations routes require the `X-API-Key` request header.

### `GET /`
Serves a premium, interactive landing page displaying the **Active Model**, **Shadow Model**, and **Database Status**.

### `POST /predict`
*   **Request Schema:**
    ```json
    {
      "customer_id": "cust_12345",
      "age": "2",
      "gender": "M",
      "category": "es_transportation",
      "amount": 25.5
    }
    ```
*   **Response:** JSON with `diagnosis`, `confidence`, and `probabilities`.

### `POST /feedback`
Receives ground truth labels to close the ML loop.
*   **Request Schema:** `{"prediction_id": 1052, "actual_fraud": 1}`

### `POST /retrain`
Queues a new model training execution via Celery workers. (Auth Required)

### `POST /reload-model?load_as_shadow=False`
Commands the service to poll ZenML and atomically swap memory pointers. 
*   If `load_as_shadow=False`, it pulls from the **`PRODUCTION`** stage and hot-swaps the active pointer immediately. 
*   If `load_as_shadow=True`, it pulls the candidate model from the **`STAGING`** stage and loads it silently as the shadow model without impacting active traffic. 
*(Auth Required)*

### `POST /promote-shadow`
Instantly swaps the loaded shadow candidate model into the active production slot. *(Auth Required)*
*   **Request Payload:** None. The API automatically swaps the model currently held in the internal `shadow_model_version` memory pointer. Send an empty request body.
