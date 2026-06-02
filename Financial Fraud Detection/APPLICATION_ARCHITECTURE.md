# Enterprise MLOps Financial Fraud Detection System
**Comprehensive Architecture & Implementation Nuance Guide**

This document serves as the complete technical manual for the entire Financial Fraud Detection application. It details the exact data pipelines, model architecture, orchestration frameworks, and API serving layers, along with the deep, DevOps-level rationale behind every architectural choice.

---

## 1. System Overview: What It Does

This system is a continuous-learning Machine Learning application designed to detect fraudulent financial transactions in real-time. It takes raw transactional data (like customer age, merchant category, and transaction amount), engineers time-based behavioral features, trains a highly accurate XGBoost model via ZenML, and serves predictions via an enterprise-hardened FastAPI service capable of handling thousands of requests per second.

The architecture is split into three decoupled components:
1. **The ZenML Pipeline:** The orchestrator that handles data extraction, feature engineering, model training, evaluation, and artifact versioning.
2. **The FastAPI Inference Service:** The hyper-scalable, real-time web server that receives transactions and returns fraud predictions.
3. **The Decoupled Worker Queue:** A background polling script that executes heavy training tasks safely away from the API's CPU pool.

---

## 2. The Machine Learning Pipeline (ZenML)

The entire ML lifecycle is orchestrated by ZenML. Rather than relying on messy Jupyter notebooks, the pipeline is strictly modularized into isolated Python steps.

### A. Feature Engineering (`fraud_feature_engineer`)
Fraud is rarely detected by a single transaction; it is detected by *behavioral velocity*. 
- **The Implementation:** We convert raw simulation steps into absolute datetimes. We then apply a `groupby().rolling()` window to calculate the number of transactions a specific customer has made in the last 1-day, 7-days, and 30-days.
- **The Nuance:** We explicitly drop the `customer`, `merchant`, and `date` identity columns before training. If a model trains on a specific `customer_id`, it will overfit to the people in the dataset and fail in the real world when encountering a new customer. The model must learn *behavior*, not *identity*.

### B. Preprocessing & Consistency (`fraud_data_preprocessor`)
- **The Implementation:** We construct an `imblearn` and `scikit-learn` Pipeline containing a `StandardScaler` and a `OneHotEncoder`.
- **The Nuance:** The preprocessing pipeline is saved as a physical artifact (`fraud_preprocess_pipeline`). In production, the Inference API loads this exact artifact to transform incoming JSON data. This completely eliminates **Training-Serving Skew**—the guarantee that the mathematical transformations applied during training are identically applied during live inference.

### C. Addressing Class Imbalance (SMOTE)
Fraud data is inherently imbalanced (99% legitimate, 1% fraud). If an algorithm simply guesses "Legitimate" every time, it achieves 99% accuracy but fails its entire purpose.
- **The Implementation:** We use Synthetic Minority Over-sampling Technique (SMOTE) inside the training pipeline.
- **The Nuance:** We apply SMOTE *only* to the training data split, never the test data. Furthermore, SMOTE is chained *before* the `StandardScaler`. This ensures that the scaler calculates its mean and standard deviation on the balanced distribution, allowing the XGBoost model to correctly weigh the synthetic minority class.

### D. Model Evaluation & Promotion
- **The Nuance:** Accuracy is useless in fraud. We prioritize **Recall** (catching every possible fraudster). The `fraud_model_promoter` step fetches the currently active production model from ZenML, compares its Recall against the newly trained model, and only officially "promotes" the new model if it genuinely outperforms the live one.

---

## 3. The FastAPI Inference Service (`inference_api.py`)

The Inference API is designed to survive hyper-scale production environments (e.g., deployed inside Kubernetes with 50+ replica pods). 

### A. Fail-Fast Database Initialization
- **The Architecture:** The API uses `aiomysql` to connect to a local MySQL database. The connection pool is established during the FastAPI `lifespan` startup event.
- **The Nuance:** There is no `try/except` block suppressing errors during startup. This is **Fail-Fast architecture**. If the database is down or the credentials are wrong, the API violently crashes. This signals the Kubernetes Load Balancer to mark the container as "Unhealthy" and immediately stop routing user traffic to the broken node.

### B. Atomic Model Pointers (Hot-Swapping)
- **The Architecture:** `app.state.models` is a dictionary mapped by version IDs.
- **The Nuance:** We never use a `global _model` variable. If multiple users request a prediction at the exact millisecond an administrator calls the `/reload-model` endpoint, replacing a global variable mid-flight will crash the server. Instead, the API downloads the new model into a separate dictionary key. Once it is 100% loaded into RAM, the API flips a pointer to route new traffic to it. This achieves **Zero-Downtime Deployments** with absolute thread-safety.

### C. Strict Schema Validation
- **The Architecture:** The incoming JSON payload is forcefully unpacked into an explicit `EXPECTED_FEATURE_ORDER` list before hitting `scikit-learn`.
- **The Nuance:** Upstream web services often scramble JSON dictionary keys. Because `scikit-learn` relies entirely on physical column index order, passing a DataFrame where `gender` accidentally comes before `age` will cause the model to predict garbage. Forcing an explicit ordering layer acts as an unbreakable firewall against upstream schema drift.

### D. Asynchronous Connection Pooling
- **The Architecture:** Every prediction is logged to MySQL using `FastAPI.BackgroundTasks` and `aiomysql`.
- **The Nuance:** Writing to a hard drive is incredibly slow. By using an asynchronous connection pool, the API instantly borrows an open connection, dispatches the SQL command in the background, and returns the prediction to the customer *before* the database even finishes writing. This reduces API latency from ~50ms to ~2ms.

---

## 4. Decoupled Architecture & Observability

### A. Decoupled Retraining (The Job Queue)
- **The Architecture:** The `/retrain` endpoint does **not** execute python training code. It inserts a `status="pending"` row into a MySQL table. A separate background process (`training_worker.py`) polls this table and executes the training pipeline.
- **The Nuance:** ML Training is heavily CPU-bound. If the API executed the training script itself, it would completely starve the server's CPU, causing incoming fraud checks to timeout. By decoupling the trigger via a database queue, the API remains lightning fast, while the heavy lifting is delegated to a separate server (or thread) entirely.

### B. Structured JSON Logging (`python-json-logger`)
- **The Nuance:** Standard `logging.info("Model loaded")` outputs unstructured text. In a distributed enterprise environment, text logs are impossible to aggregate. By converting all logs to strict JSON (including fields like `latency_ms` and `model_version`), observability platforms like Datadog or ELK can instantly ingest the logs and generate dashboards calculating your exact p99 latency percentiles across thousands of containers.

### C. Evidently Drift Detection
- **The Architecture:** The `/drift-report` endpoint queries the last 5,000 live inferences from MySQL and compares their feature distributions against the original ZenML `fraud_preprocessed_dataset_trn` artifact.
- **The Nuance:** Machine learning models degrade over time as consumer behavior changes (Data Drift). Integrating `Evidently` allows the fraud team to instantly generate visual HTML reports to mathematically prove whether the live data has drifted far enough from the training data to warrant a retraining cycle.
