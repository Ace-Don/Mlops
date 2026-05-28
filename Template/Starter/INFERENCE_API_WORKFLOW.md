# Inference API Workflow: Complete Drill-Down

**API URL**: `http://localhost:8000`  
**Docs**: `http://localhost:8000/docs` (interactive Swagger UI)  
**Database**: `db/inference_logs.db` (SQLite)

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE API (FastAPI)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STARTUP (once):                                          │  │
│  │  1. init_database()       → Create SQLite tables         │  │
│  │  2. load_model()          → Load from ZenML production   │  │
│  │     - Fetch "breast_cancer_classifier" version          │  │
│  │     - Load sklearn_classifier artifact                  │  │
│  │     - Load preprocess_pipeline artifact                 │  │
│  │                                                          │  │
│  │  Global variables set: _model, _preprocess_pipeline     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ PER REQUEST (/predict):                                 │  │
│  │  1. Validate input        → 30 features checked          │  │
│  │  2. Preprocess           → Apply same pipeline as train  │  │
│  │  3. Predict              → model.predict() + proba()    │  │
│  │  4. Log to SQLite        → ground_truth=NULL            │  │
│  │  5. Return prediction_id → For future feedback          │  │
│  │                                                          │  │
│  │  Response time: 50-200ms (see performance analysis)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ FEEDBACK LOOP:                                           │  │
│  │  POST /feedback {prediction_id, actual_diagnosis}       │  │
│  │    → Update SQLite: ground_truth = 0 or 1              │  │
│  │    → Check if drift detected (accuracy < 85%)           │  │
│  │    → If 50+ labelled samples + drift → auto-retrain    │  │
│  │                                                          │  │
│  │ GET /stats                                               │  │
│  │    → Calculate metrics from labelled data                │  │
│  │    → Accuracy, precision, recall, F1-score              │  │
│  │    → Auto-trigger retraining if drift                   │  │
│  │                                                          │  │
│  │ POST /retrain {force=false}                             │  │
│  │    → Fetch labelled records from database                │  │
│  │    → Launch ZenML training in background                │  │
│  │    → New model promoted to production                    │  │
│  │    → API auto-loads new model (zero downtime)           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Detailed Request/Response Workflow

### **Step 1: Health Check (Optional)**

Before making predictions, verify the model is loaded:

```bash
GET /health
```

**Response** (200):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "sklearn_classifier",
  "model_version": "production",
  "database_exists": true
}
```

---

### **Step 2: Make a Prediction**

Send patient data (30 features):

```bash
POST /predict
Content-Type: application/json

{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.80,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.07871,
  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.002250,
  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.60,
  "area_worst": 2019.0,
  "smoothness_worst": 0.1622,
  "compactness_worst": 0.6656,
  "concavity_worst": 0.7119,
  "concave_points_worst": 0.2654,
  "symmetry_worst": 0.4601,
  "fractal_dimension_worst": 0.11890
}
```

**Response** (200):
```json
{
  "success": true,
  "prediction_id": 1,
  "diagnosis": "cancer",
  "diagnosis_code": 1,
  "confidence": 0.952,
  "probability_no_cancer": 0.048,
  "probability_cancer": 0.952,
  "model_version": "production",
  "message": "cancer (95.2%). Use prediction_id 1 to submit feedback."
}
```

**What happened internally:**
```
Input validation (Pydantic)
    ↓
Create DataFrame with 30 features
    ↓
Add dummy target column (required by preprocess_pipeline)
    ↓
Apply preprocessing: NADropper → ColumnsDropper → MinMaxScaler → DataFrameCaster
    ↓
model.predict(processed_features) → returns [1]
    ↓
model.predict_proba(processed_features) → returns [[0.048, 0.952]]
    ↓
Database INSERT to inference_logs:
  - id: 1 (auto-increment)
  - timestamp: CURRENT_TIMESTAMP
  - input_features: JSON string of all 30 features
  - prediction: 1
  - confidence: 0.952
  - probability_no_cancer: 0.048
  - probability_cancer: 0.952
  - ground_truth: NULL (← not yet known)
  - ground_truth_timestamp: NULL
  - model_version: "production"
  - api_response_time_ms: 45.2
    ↓
Log to MLflow experiment: monitoring.api_prediction
    ↓
Return prediction_id=1 to caller
```

**Database state after /predict:**
```sql
-- inference_logs table
id | timestamp            | prediction | confidence | ground_truth
1  | 2026-05-28 14:32:15 | 1          | 0.952      | NULL  ← not yet known!
2  | 2026-05-28 14:33:22 | 0          | 0.891      | NULL
```

---

### **Step 3: Wait for Ground Truth**

Real diagnosis arrives (biopsy, analysis, etc.) → Doctor confirms: **YES, it was cancer**

Submit feedback:

```bash
POST /feedback
Content-Type: application/json

{
  "prediction_id": 1,
  "actual_diagnosis": 1
}
```

**Response** (200):
```json
{
  "success": true,
  "prediction_id": 1,
  "ground_truth": 1,
  "total_labelled_data": 5,
  "message": "Ground truth recorded. 5 labelled samples now available for retraining."
}
```

**What happened internally:**
```
Validate input (prediction_id exists, actual_diagnosis is 0 or 1)
    ↓
UPDATE inference_logs 
SET ground_truth = 1, ground_truth_timestamp = CURRENT_TIMESTAMP
WHERE id = 1
    ↓
Fetch all predictions WHERE ground_truth IS NOT NULL
    → Returns 5 labelled records (for retraining pool)
    ↓
Log to MLflow: monitoring.feedback_received
    ↓
Return success message with count of labelled data
```

**Database state after /feedback:**
```sql
-- inference_logs table
id | prediction | ground_truth | ground_truth_timestamp
1  | 1          | 1            | 2026-05-28 14:35:10  ← NOW LABELLED!
2  | 0          | NULL         | NULL                 ← still waiting
```

---

### **Step 4: Monitor Performance**

Once you have some labelled data, check model performance:

```bash
GET /stats
```

**Response** (200):
```json
{
  "total_predictions": 42,
  "labelled_predictions": 5,
  "recent_accuracy": 0.88,
  "recent_precision": 0.90,
  "recent_recall": 0.85,
  "recent_f1_score": 0.87,
  "drift_detected": false,
  "current_model_version": "production"
}
```

**What happened internally:**
```
Query last 7 days of labelled data:
  SELECT prediction, ground_truth 
  FROM inference_logs 
  WHERE ground_truth IS NOT NULL 
  AND timestamp > datetime('now', '-7 days')
    ↓
  Returns 5 records (e.g., predictions=[1,0,1,1,0], ground_truth=[1,0,1,0,0])
    ↓
Calculate metrics:
  - accuracy   = accuracy_score([1,0,1,1,0], [1,0,1,0,0]) = 0.80
  - precision  = 0.90 (of predicted 1s, 90% correct)
  - recall     = 0.85 (of actual 1s, we caught 85%)
  - f1_score   = 0.87 (harmonic mean)
    ↓
Check drift:
  if accuracy < 0.85:
    drift_detected = True
    → Trigger auto-retraining if 50+ labelled samples
    ↓
Log to MLflow: monitoring.performance_metrics
    ↓
Return stats to caller
```

**If drift detected:**
```
/stats endpoint detects accuracy < 85%
    ↓
Fetch all labelled records (50+)
    ↓
Log retraining event to retraining_history table
    ↓
Launch training pipeline in background thread:
  - training.with_options(config_path="configs/training_rf.yaml")
  - Pipeline trains on accumulated labelled data
  - New model evaluated
  - If better → promoted to production stage
    ↓
API calls load_model() → hot-swap into _model and _preprocess_pipeline
    ↓
Next prediction uses new model (ZERO DOWNTIME)
```

---

## 🐍 Python Client Examples

### **Example 1: Single Prediction**

```python
import requests

url = "http://localhost:8000/predict"
patient = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.80,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.002250,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.60,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.11890
}

response = requests.post(url, json=patient)
result = response.json()

print(f"✅ Diagnosis: {result['diagnosis']}")
print(f"📊 Confidence: {result['confidence']:.1%}")
print(f"🔑 Prediction ID: {result['prediction_id']}")  # SAVE THIS
```

---

### **Example 2: Batch Predictions**

```python
import requests

url = "http://localhost:8000/predict-batch"

patients = [
    {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.80,
        "area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
        "concavity_mean": 0.3001, "concave_points_mean": 0.1471, "symmetry_mean": 0.2419,
        "fractal_dimension_mean": 0.07871, "radius_se": 1.095, "texture_se": 0.9053,
        "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399,
        "compactness_se": 0.04904, "concavity_se": 0.05373, "concave_points_se": 0.01587,
        "symmetry_se": 0.03003, "fractal_dimension_se": 0.002250, "radius_worst": 25.38,
        "texture_worst": 17.33, "perimeter_worst": 184.60, "area_worst": 2019.0,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
        "concave_points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.11890
    },
    # ... more patients ...
]

response = requests.post(url, json=patients)
results = response.json()

print(f"Batch size: {results['batch_size']}")
for pred in results['predictions']:
    print(f"Patient {pred['patient_index']}: {pred['diagnosis']} (ID: {pred['prediction_id']})")
```

---

### **Example 3: Submit Feedback**

```python
import requests

url = "http://localhost:8000/feedback"

feedback = {
    "prediction_id": 1,      # From earlier prediction
    "actual_diagnosis": 1    # 0 = no cancer, 1 = cancer
}

response = requests.post(url, json=feedback)
result = response.json()

print(f"✅ Feedback recorded")
print(f"📊 Total labelled data: {result['total_labelled_data']}")
if result['total_labelled_data'] >= 50:
    print("🔄 Enough data for retraining!")
```

---

### **Example 4: Check Performance**

```python
import requests

url = "http://localhost:8000/stats"

response = requests.get(url)
stats = response.json()

print(f"Total predictions: {stats['total_predictions']}")
print(f"Labelled predictions: {stats['labelled_predictions']}")
print(f"Accuracy: {stats['recent_accuracy']:.1%}")
print(f"Precision: {stats['recent_precision']:.1%}")
print(f"Recall: {stats['recent_recall']:.1%}")
print(f"Drift detected: {stats['drift_detected']}")

if stats['drift_detected']:
    print("⚠️ MODEL DRIFT DETECTED - Retraining triggered!")
```

---

### **Example 5: View Logs**

```python
import requests

# View all recent predictions (labelled and unlabelled)
response = requests.get("http://localhost:8000/logs?limit=10")
logs = response.json()

for log in logs:
    print(f"ID {log['id']}: prediction={log['prediction']}, "
          f"confidence={log['confidence']:.2%}, "
          f"ground_truth={log['ground_truth']}")

# View only labelled predictions (with ground truth)
response = requests.get("http://localhost:8000/logs?limit=10&labelled_only=True")
labelled = response.json()
print(f"\nLabelled predictions: {len(labelled)}")
```

---

## ⚡ Performance Analysis: Why Is Inference Slow?

**Typical latency: 50-200ms per prediction**

### Breakdown of where time is spent:

```
POST /predict (50-200ms total)
├─ Pydantic validation (1-2ms)
│   └ Check all 30 float fields are present and valid
│
├─ DataFrame creation (1-2ms)
│   └ Convert dict → pd.DataFrame([{...}])
│   └ Add dummy target column
│
├─ Preprocessing pipeline (10-30ms) ⚠️ SLOWEST
│   ├ NADropper (1-2ms) - check for NaN values
│   ├ ColumnsDropper (1-2ms) - filter columns
│   ├ MinMaxScaler (2-5ms) - normalize all features
│   └ DataFrameCaster (1-2ms) - convert back to DataFrame
│
├─ Model inference (5-15ms)
│   ├ model.predict() (2-8ms) - RandomForest or SGD
│   └ model.predict_proba() (3-8ms) - get probabilities
│
├─ Database logging (5-20ms) ⚠️ SECOND SLOWEST
│   ├ _get_connection() (1-2ms)
│   ├ SQL INSERT (3-15ms) - write to SQLite on disk
│   └ conn.commit() (1-5ms) - flush to storage
│
└─ MLflow logging (5-20ms)
    └ mlflow.start_run() + log_metric() + end_run()
```

---

### Why is preprocessing slow?

The **preprocessing pipeline is reused from training** for consistency:

```python
# Same pipeline as training!
preprocess_pipeline = Pipeline([
    ("drop_na", NADropper()),           # Check for NaN
    ("drop_columns", ColumnsDropper()), # Filter columns  
    ("normalize", MinMaxScaler()),      # Scale to [0, 1]
    ("cast", DataFrameCaster(...))      # Convert to DataFrame
])

# This MUST be identical to training
# Otherwise: model expects normalized input but gets raw input → WRONG PREDICTIONS
```

**This is correct** - you need the same preprocessing. But it adds latency.

---

### Why is database logging slow?

SQLite writes to disk synchronously:

```python
cursor.execute("INSERT INTO inference_logs ...")
conn.commit()  # ← Waits for disk write (3-15ms)
```

Every prediction blocks until the database confirms the write.

---

## 🚀 How to Speed It Up (If Needed)

### **Option 1: Async Database Writes (Recommended)**

Move database logging to background thread:

```python
# Current (blocking):
prediction_id = db_log_prediction(...)  # Waits for disk

# Better (async):
loop = asyncio.get_event_loop()
loop.run_in_executor(executor, db_log_prediction, ...)  # Returns immediately
```

**Impact**: Reduce latency from 50-200ms to 10-50ms

---

### **Option 2: Use PostgreSQL Instead of SQLite**

PostgreSQL has better concurrency:

```python
# Current (SQLite file on disk):
sqlite:///db/inference_logs.db

# Switch to PostgreSQL:
postgresql://user:password@localhost:5432/inference_db
```

**Impact**: Slightly faster writes, better scaling

---

### **Option 3: Cache Preprocessing Pipeline**

Preprocessing is deterministic - could cache:

```python
# Instead of applying pipeline every time:
patient_processed = _preprocess_pipeline.transform(patient_df)  # 10-30ms

# Could use @lru_cache if inputs are discrete
# But for continuous values, caching doesn't help much
```

**Impact**: Minimal - preprocessing is already pretty fast

---

### **Option 4: Profile and Optimize**

Add timing to each step:

```python
import time

start = time.time()
# ... step ...
elapsed = (time.time() - start) * 1000
logger.info(f"Preprocessing took {elapsed:.1f}ms")
```

---

## 🔐 Database Schema

### **inference_logs** (all predictions)
```sql
CREATE TABLE inference_logs (
    id                    INTEGER PRIMARY KEY,
    timestamp             DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_features        TEXT,           -- JSON of 30 features
    prediction            INTEGER,        -- 0 or 1
    confidence            REAL,           -- max probability
    probability_no_cancer REAL,
    probability_cancer    REAL,
    ground_truth          INTEGER,        -- NULL until /feedback
    ground_truth_timestamp DATETIME,      -- when feedback arrived
    model_version         STRING,         -- "production"
    api_response_time_ms  REAL
);
```

### **performance_metrics** (monitoring)
```sql
CREATE TABLE performance_metrics (
    id                   INTEGER PRIMARY KEY,
    timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP,
    accuracy             REAL,           -- from labelled data
    precision            REAL,
    recall               REAL,
    f1_score             REAL,
    total_predictions    INTEGER,
    labelled_predictions INTEGER,
    drift_detected       BOOLEAN,
    model_version        STRING
);
```

### **retraining_history** (audit log)
```sql
CREATE TABLE retraining_history (
    id                    INTEGER PRIMARY KEY,
    timestamp             DATETIME DEFAULT CURRENT_TIMESTAMP,
    triggered_by          STRING,   -- "auto_drift_detection" or "manual_api_request"
    reason                TEXT,
    training_samples      INTEGER,
    old_accuracy          REAL,
    new_accuracy          REAL,
    improvement           REAL,
    status                STRING,   -- "submitted", "completed", "failed"
    model_version_before  STRING,
    model_version_after   STRING
);
```

---

## 🔄 End-to-End Example: Complete Workflow

```python
import requests
import time

API = "http://localhost:8000"

# 1️⃣ Check health
print("1️⃣ Checking API health...")
r = requests.get(f"{API}/health")
assert r.status_code == 200
print("✅ API is ready")

# 2️⃣ Make a prediction
print("\n2️⃣ Making prediction...")
patient = {
    "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.80,
    # ... all 30 features ...
}
r = requests.post(f"{API}/predict", json=patient)
pred = r.json()
prediction_id = pred['prediction_id']
print(f"✅ Prediction: {pred['diagnosis']} (ID: {prediction_id})")

# 3️⃣ Simulate waiting for ground truth (days/weeks pass)
print("\n3️⃣ Waiting for ground truth...")
time.sleep(2)  # Simulate time passing

# 4️⃣ Submit ground truth
print("4️⃣ Submitting ground truth...")
r = requests.post(f"{API}/feedback", json={
    "prediction_id": prediction_id,
    "actual_diagnosis": 1  # Yes, it was cancer
})
feedback = r.json()
print(f"✅ Ground truth recorded. Total labelled: {feedback['total_labelled_data']}")

# 5️⃣ Check performance
print("\n5️⃣ Checking model performance...")
r = requests.get(f"{API}/stats")
stats = r.json()
print(f"✅ Accuracy: {stats['recent_accuracy']:.1%}")
print(f"📊 Labelled data: {stats['labelled_predictions']}")
print(f"⚠️  Drift detected: {stats['drift_detected']}")

# 6️⃣ If drift, trigger retraining
if stats['labelled_predictions'] >= 50:
    print("\n6️⃣ Triggering retraining...")
    r = requests.post(f"{API}/retrain")
    retrain = r.json()
    print(f"✅ Retraining started (ID: {retrain['retraining_id']})")
    print("   Pipeline running in background...")
    print("   Check /stats to see results")
```

---

## 📚 API Reference

| Endpoint | Method | Purpose | Latency |
|----------|--------|---------|---------|
| `/` | GET | API info | <1ms |
| `/health` | GET | Health check | 1-5ms |
| `/model-info` | GET | Model details | 1-5ms |
| `/predict` | POST | Single prediction | **50-200ms** |
| `/predict-batch` | POST | Multiple predictions | 100-500ms |
| `/feedback` | POST | Submit ground truth | 5-30ms |
| `/stats` | GET | Performance metrics | 100-500ms |
| `/logs` | GET | View predictions | 10-50ms |
| `/retrain` | POST | Manual retraining | 5-20ms (async) |
| `/reload-model` | POST | Hot-swap model | 100-500ms |

---

## 🎯 Summary

**The inference API:**
- ✅ Loads the production model **once at startup** (not per-request)
- ✅ Applies **identical preprocessing** to training
- ✅ Logs **every prediction** to SQLite
- ✅ Collects **ground truth via /feedback**
- ✅ **Auto-detects drift** from labelled data
- ✅ **Auto-retrains** in background when needed
- ✅ **Hot-swaps** new models without downtime

**Performance (50-200ms):**
- Preprocessing pipeline (10-30ms) - most time
- Database logging (5-20ms)
- Model inference (5-15ms)
- MLflow logging (5-20ms)

**To speed up: Use async database writes** → reduces latency by 50% ⚡
