# 🔍 MLflow Integration Guide

Complete documentation of MLflow logging integration across your ZenML MLOps pipeline.

---

## Overview

MLflow has been seamlessly integrated into your entire pipeline to track:
- **Parameters** - Settings used (model type, test size, normalization, etc.)
- **Metrics** - Performance measurements (accuracy, precision, recall, F1)
- **Models** - Trained model artifacts
- **Datasets** - Information about data flowing through each step

This works **alongside ZenML's tracking** - both systems now monitor your pipelines, giving you the best of both worlds.

---

## What Changed?

### Updated Files

1. **requirements.txt** - Added `mlflow>=2.0.0`
2. **steps/data_preprocessor.py** - Logs preprocessing parameters and dataset shapes
3. **steps/data_splitter.py** - Logs split ratios and dataset sizes
4. **steps/model_trainer.py** - Logs model type, hyperparameters, and trained model artifact
5. **steps/model_evaluator.py** - Logs all evaluation metrics (accuracy, precision, recall, F1)
6. **steps/model_promoter.py** - Logs promotion decisions and comparisons
7. **steps/inference_preprocessor.py** - Logs inference dataset info
8. **steps/inference_predict.py** - Logs prediction statistics and confidence scores

---

## MLflow Experiments Structure

### Three Main Experiments

```
feature_engineering
├── data_preprocessor run
│   ├── Params: drop_na, normalize, drop_columns, random_state
│   └── Metrics: train_dataset_rows, test_dataset_rows, train_dataset_cols
└── data_splitter run
    ├── Params: test_size, train_size, random_state
    └── Metrics: total_rows, train_rows, test_rows, train_percentage

model_training
├── model_trainer run
│   ├── Params: model_type, target, [model-specific params]
│   ├── Metrics: training_samples, training_features
│   └── Artifacts: sklearn_classifier (trained model)
├── model_evaluator run
│   ├── Params: min_train_accuracy, min_test_accuracy, target
│   └── Metrics: train_accuracy, test_accuracy, precision, recall, f1_score
└── model_promoter run
    ├── Params: stage, accuracy, accuracy_threshold, promotion_status
    ├── Metrics: previous_accuracy (if comparing)
    └── Decision: promoted or rejected

inference
├── inference_preprocessor run
│   ├── Params: target_column
│   └── Metrics: inference_samples, inference_features, preprocessed_features
└── inference_predict run
    ├── Metrics: inference_samples, average_prediction_confidence
    └── Statistics: class_0_count, class_1_count
```

---

## What Gets Logged

### data_preprocessor.py

**Parameters:**
```python
drop_na: true/false              # Whether to remove rows with missing values
normalize: true/false            # Whether to scale numbers 0-1
drop_columns: [col1, col2, ...]  # Columns to remove
random_state: 17                 # Random seed for reproducibility
target: "target"                 # Target column name
```

**Metrics:**
```python
train_dataset_rows: 455          # Number of training samples
test_dataset_rows: 114           # Number of test samples
train_dataset_cols: 31           # Number of columns before splitting
```

**Why it matters:**
Answers "What preprocessing settings were used?" and "How much data did we have at each stage?"

---

### data_splitter.py

**Parameters:**
```python
test_size: 0.2                   # 20% for testing
train_size: 0.8                  # 80% for training
random_state: 42                 # Ensures reproducible splits
```

**Metrics:**
```python
total_rows: 569                  # Total dataset size
train_rows: 455                  # Samples in training set
test_rows: 114                   # Samples in test set
train_percentage: 80.0           # Percentage verification
```

**Why it matters:**
Confirms the data split and tracks dataset sizes through the pipeline.

---

### model_trainer.py

**Parameters:**
```python
model_type: "sgd" or "rf"        # Algorithm chosen
target: "target"                 # Target column
# Algorithm-specific:
loss: "hinge"                    # (SGD only) Loss function
penalty: "l2"                    # (SGD only) Regularization
n_estimators: 100                # (RF only) Number of trees
max_depth: null                  # (RF only) Tree depth limit
```

**Metrics:**
```python
training_samples: 455            # Number of training examples
training_features: 30            # Number of input features
```

**Artifacts:**
```
sklearn_classifier               # The trained model saved to MLflow
```

**Why it matters:**
Tracks which model and hyperparameters were used, saves the actual trained model for later reference/deployment.

---

### model_evaluator.py

**Parameters:**
```python
min_train_accuracy: 0.0          # Minimum acceptable training accuracy
min_test_accuracy: 0.0           # Minimum acceptable test accuracy
target: "target"                 # Target column
```

**Metrics:**
```python
train_accuracy: 0.956            # Training set accuracy (95.6%)
test_accuracy: 0.947             # Test set accuracy (94.7%)
test_precision: 0.965            # Of predicted positives, how many correct?
test_recall: 0.940               # Of actual positives, how many found?
test_f1_score: 0.952             # Harmonic mean of precision & recall
```

**Why it matters:**
Complete picture of model performance. If test_accuracy is 94.7%, precision is 96.5%, and recall is 94%, you know the model is well-balanced.

---

### model_promoter.py

**Parameters:**
```python
stage: "production"              # Which stage to promote to
accuracy: 0.947                  # Current model's accuracy
accuracy_threshold: 0.8          # Minimum required (80%)
promotion_status: string         # One of:
  # "rejected_low_accuracy" - Below threshold
  # "promoted_first_model" - First time, no predecessor
  # "promoted_better_accuracy" - Better than previous
  # "rejected_worse_accuracy" - Worse than previous
```

**Metrics (when comparing):**
```python
previous_accuracy: 0.936         # Last production model's accuracy
```

**Why it matters:**
Shows exactly why a model was promoted or rejected. You can see the audit trail of which models went to production and why.

---

### inference_preprocessor.py

**Parameters:**
```python
target_column: "target"          # Temporary target column added
```

**Metrics:**
```python
inference_samples: 50            # Number of new patients to predict
inference_features: 31           # Features in raw data
preprocessed_features: 30        # Features after preprocessing
```

**Why it matters:**
Verifies that new inference data is preprocessed correctly (same number of features as training).

---

### inference_predict.py

**Metrics:**
```python
inference_samples: 50            # Number of predictions made
average_prediction_confidence: 0.92  # How confident is the model? (0-1 scale)
class_0_count: 15                # Number predicted as "no cancer"
class_1_count: 35                # Number predicted as "cancer"
```

**Why it matters:**
Shows what predictions were made, distribution across classes, and model confidence in real-time.

---

## How to Use MLflow

### Start MLflow UI

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

### View Your Experiments

Navigate through:
- **feature_engineering** - See how data was prepared
- **model_training** - See all model runs with metrics
- **inference** - See real-time predictions

### Compare Models

In MLflow UI:
1. Go to **model_training** experiment
2. Select multiple runs with checkboxes
3. Click "Compare" to see side-by-side metrics
4. Identify which model type (SGD vs Random Forest) performs better

### Track Parameters vs Metrics

- **Parameters** (left side) - Settings that don't change during training
- **Metrics** (right side) - Values that improve during training/evaluation
- **Artifacts** (bottom) - Models and files saved

---

## Complete Data Flow with MLflow

```
DAY 1: Feature Engineering

Run.py --feature-pipeline
  ↓
data_loader() 
  → Loads 569 breast cancer records
  ↓
data_preprocessor()
  → MLflow logs:
    ✓ Params: drop_na=true, normalize=true
    ✓ Metrics: 569 total rows
  ↓
data_splitter()
  → MLflow logs:
    ✓ Params: test_size=0.2
    ✓ Metrics: 455 train rows, 114 test rows
  ↓
Artifacts stored in ZenML + MLflow tracks parameters
```

```
DAY 2: Model Training

Run.py --training-pipeline
  ↓
feature_engineering() [cached from Day 1]
  ↓
model_trainer()
  → MLflow logs:
    ✓ Params: model_type=rf, n_estimators=100
    ✓ Metrics: 455 training samples
    ✓ Artifacts: trained Random Forest model
  ↓
model_evaluator()
  → MLflow logs:
    ✓ Metrics: train_accuracy=95.6%, test_accuracy=94.7%
    ✓ Metrics: precision=96.5%, recall=94.0%, f1=95.2%
  ↓
model_promoter()
  → MLflow logs:
    ✓ Params: promotion_status=promoted_first_model
    ✓ Model moved to "production" stage in ZenML
  ↓
MLflow UI shows: Random Forest trained, 94.7% accurate, promoted to production
```

```
DAY 3: Make Predictions

Run.py --inference-pipeline
  ↓
inference_preprocessor()
  → MLflow logs:
    ✓ Metrics: 50 new patients to predict
    ✓ Metrics: preprocessed_features=30
  ↓
inference_predict()
  → MLflow logs:
    ✓ Metrics: average_prediction_confidence=89%
    ✓ Metrics: 35 predicted "cancer", 15 predicted "no cancer"
  ↓
Predictions returned to doctors with confidence scores
MLflow shows what predictions were made and how confident the model was
```

---

## Common MLflow Queries

### "Which model had the best accuracy?"
1. Go to MLflow UI → model_training experiment
2. Sort by **test_accuracy** (descending)
3. Top row = best model

### "Did normalization help?"
1. Compare two runs: one with normalize=true, one with normalize=false
2. Check if test_accuracy improved
3. MLflow shows the difference

### "How confident are my predictions?"
1. Go to inference experiment
2. Find average_prediction_confidence metric
3. 0.95 = very confident, 0.51 = barely confident

### "Why was a model rejected?"
1. Go to model_training experiment
2. Look at model_promoter run
3. promotion_status = "rejected_low_accuracy" or "rejected_worse_accuracy"
4. Check accuracy metric to understand why

---

## Integration with ZenML (Side-by-Side)

**ZenML tracks:**
- Complete pipeline lineage and run history
- All artifacts and versions
- Connections between steps
- Infrastructure/stack details

**MLflow tracks:**
- Experiment organization
- Model comparison across runs
- Hyperparameter tuning
- Model registry and staging

**Together they provide:**
- Full reproducibility (ZenML lineage + MLflow params)
- Easy comparison (MLflow experiments)
- Production ready (ZenML stages + MLflow registry)
- Audit trail (both systems track everything)

---

## Best Practices

### 1. Use Consistent Experiment Names
Already done! Each pipeline stage has its experiment:
- `feature_engineering`
- `model_training`
- `inference`

### 2. Log at the Right Granularity
✅ **Do:** Log model parameters before training
```python
mlflow.log_param("model_type", model_type)
```

❌ **Don't:** Log inside a loop (creates hundreds of logs)

### 3. Use Descriptive Metric Names
✅ **Good:** `test_accuracy`, `precision`, `f1_score`
❌ **Bad:** `acc1`, `m1`, `val2`

### 4. Compare Experiments Regularly
Use MLflow UI to see which configurations work best before deploying.

### 5. Archive Old Runs
Keep MLflow clean by setting old models to "archived" stage when replaced.

---

## Troubleshooting

### "MLflow not tracking anything"
**Check:**
- Is MLflow installed? `pip install mlflow`
- Is mlflow ui running? `mlflow ui`
- Are mlflow.log_* calls actually in your code? ✅ They are!

### "Metrics appear but no parameters"
**Check:**
- Parameters are logged with `mlflow.log_param()`
- Already implemented in all steps ✅

### "Models not showing in artifacts"
**Check:**
- Only model_trainer saves the model (expected)
- Inference uses that saved model
- If you want model after inference, add `mlflow.sklearn.log_model()`

### "Runs appear empty"
**Check:**
- `mlflow.start_run()` before logging
- `mlflow.end_run()` after logging
- Already done in all steps ✅

---

## What's Next?

### Advanced MLflow Features (Optional)

1. **Model Registry** - Register models centrally
```python
mlflow.register_model(f"runs:{run_id}/sklearn_classifier", "cancer_detector")
```

2. **Auto Logging** - Let MLflow capture everything automatically
```python
mlflow.sklearn.autolog()
# Now model_trainer logs everything without manual calls
```

3. **Remote Tracking** - Store runs on a server instead of local
```python
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

4. **Custom Metrics** - Track domain-specific metrics
```python
mlflow.log_metric("false_positives", fp_count)
mlflow.log_metric("false_negatives", fn_count)
```

---

## Summary

✅ **MLflow is now fully integrated**
- All parameters logged before step execution
- All metrics logged after processing
- Models saved to MLflow artifact store
- Three organized experiments tracking the full ML lifecycle
- Combines with ZenML for production-grade MLOps

✅ **You can now:**
- Compare models in MLflow UI
- See parameter impact on metrics
- Track prediction confidence
- View complete audit trail of which models went to production
- Reproduce any historical result

✅ **Your system now has:**
- Version control (ZenML) + Experiment tracking (MLflow)
- Reliability + Transparency
- Reproducibility + Auditability
- Production-ready + Development-friendly

🚀 **Run your pipelines and watch MLflow track everything in real-time!**

---

*For more info: https://mlflow.org/docs/latest/index.html*
