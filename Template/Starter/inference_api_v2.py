"""
Production Inference API with Continuous Learning Loop v2.1
=============================================================

Complete production-ready FastAPI with feedback loop for continuous model improvement.

Architecture:
    1. Prediction Serving:    Load model once at startup, reuse for every request
    2. Prediction Logging:    Store all predictions to SQLite with input + output
    3. Feedback Collection:   /feedback endpoint to submit ground truth
    4. Performance Monitoring: Calculate metrics from labelled predictions
    5. Drift Detection:       Auto-detect when model performance degrades
    6. Auto-Retraining:       Trigger retraining when drift detected (runs async)
    7. Zero-Downtime Swap:    New model replaces old without restarting the API

Key Endpoints:
    POST  /predict              → Single prediction + logging
    POST  /predict-batch        → Multiple predictions
    POST  /feedback             → Submit ground truth
    POST  /retrain              → Trigger retraining
    POST  /reload-model         → Hot-swap model after retraining
    GET   /stats                → Performance metrics + auto drift trigger
    GET   /logs                 → View prediction logs
    GET   /health               → Health check
    GET   /model-info           → Model details

Continuous Learning Loop:
    /predict (ground_truth=NULL)
        → /feedback (fills ground truth)
        → /stats   (detects drift)
        → /retrain (fires in background)
        → ZenML promotes new model
        → /reload-model (zero-downtime swap)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import pandas as pd
import mlflow
import numpy as np
from zenml.client import Client
import logging
import datetime
import json
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from zenml.enums import ModelStages


# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Detector API",
    description="Production inference API with continuous learning feedback loop",
    version="2.1.0"
)

# Initialize ZenML client
client = Client()

# Thread pool for background tasks (retraining runs here so API stays responsive)
executor = ThreadPoolExecutor(max_workers=2)

# Global variables — loaded once at startup, reused on every request
_model = None
_preprocess_pipeline = None


# ============================================================================
# DATABASE ABSTRACTION LAYER
# ============================================================================
# All SQLite logic lives here. To swap to PostgreSQL or Supabase later,
# only this section needs to change — nothing else in the API touches
# the database directly.

import sqlite3

_db_path = "db/inference_logs.db"


def _get_connection():
    """
    Return a live database connection.
    Central point for connection management — swap this to change DB backend.
    """
    Path(_db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(_db_path)


def init_database() -> bool:
    """
    Initialize all database tables on first run.

    Tables:
    - inference_logs:       Every prediction + input + ground truth (NULL until feedback arrives)
    - performance_metrics:  Rolling model performance snapshots over time
    - retraining_history:   Audit log of every retraining event triggered
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        # Main inference logs table
        # ground_truth starts as NULL — filled in later via /feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_logs (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp               DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_features          TEXT NOT NULL,
                prediction              INTEGER NOT NULL,
                confidence              REAL NOT NULL,
                probability_no_cancer   REAL,
                probability_cancer      REAL,
                ground_truth            INTEGER,
                ground_truth_timestamp  DATETIME,
                model_version           STRING,
                api_response_time_ms    REAL
            )
        """)

        # Rolling performance metrics — one row per monitoring check
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP,
                accuracy             REAL,
                precision            REAL,
                recall               REAL,
                f1_score             REAL,
                total_predictions    INTEGER,
                labelled_predictions INTEGER,
                drift_detected       BOOLEAN,
                model_version        STRING
            )
        """)

        # Retraining audit log — track every retraining event and its outcome
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_history (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             DATETIME DEFAULT CURRENT_TIMESTAMP,
                triggered_by          STRING,
                reason                TEXT,
                training_samples      INTEGER,
                old_accuracy          REAL,
                new_accuracy          REAL,
                improvement           REAL,
                status                STRING,
                model_version_before  STRING,
                model_version_after   STRING
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        return False


def db_log_prediction(
    input_features: Dict,
    prediction: int,
    confidence: float,
    probability_no_cancer: float,
    probability_cancer: float,
    model_version: str,
    response_time_ms: float
) -> int:
    """
    Write a new prediction record to the database.

    ground_truth is NULL at this point — it gets filled in later when
    the user submits feedback via /feedback.

    Returns:
        prediction_id — pass this back to the user so they can submit feedback later
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO inference_logs
            (input_features, prediction, confidence, probability_no_cancer,
             probability_cancer, model_version, api_response_time_ms, ground_truth)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            json.dumps(input_features),
            prediction,
            confidence,
            probability_no_cancer,
            probability_cancer,
            model_version,
            response_time_ms,
            None  # ground_truth — NULL until feedback arrives
        ))

        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()

        logger.info(f"✅ Logged prediction {prediction_id}")
        return prediction_id

    except Exception as e:
        logger.error(f"❌ Failed to log prediction: {str(e)}")
        raise


def db_update_ground_truth(prediction_id: int, actual_diagnosis: int) -> bool:
    """
    Fill in the ground truth for an existing prediction.

    This is what closes the feedback loop:
    1. Prediction made → ground_truth=NULL
    2. Reality reveals the answer (biopsy result, security analyst review, etc.)
    3. User calls /feedback → this function fills in the real label
    4. Record is now labelled and usable for drift detection and retraining
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE inference_logs
            SET ground_truth = ?, ground_truth_timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (actual_diagnosis, prediction_id))

        conn.commit()
        conn.close()

        logger.info(f"✅ Ground truth updated for prediction {prediction_id}: {actual_diagnosis}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to update ground truth: {str(e)}")
        return False


def db_fetch_labelled_records() -> pd.DataFrame:
    """
    Fetch all predictions that have a ground truth label.

    Used for:
    - Calculating live performance metrics
    - Detecting model drift
    - Building the retraining dataset
    """
    try:
        conn = _get_connection()
        query = """
            SELECT id, input_features, prediction, ground_truth, timestamp
            FROM inference_logs
            WHERE ground_truth IS NOT NULL
            ORDER BY timestamp DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"✅ Fetched {len(df)} labelled records")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch labelled records: {str(e)}")
        return pd.DataFrame()


def db_fetch_recent_logs(limit: int = 20, labelled_only: bool = False) -> List[Dict]:
    """
    Fetch the most recent prediction records for the /logs endpoint.

    Args:
        limit:         How many records to return
        labelled_only: If True, only return records with ground truth
    """
    try:
        conn = _get_connection()
        where = "WHERE ground_truth IS NOT NULL" if labelled_only else ""
        query = f"""
            SELECT id, timestamp, prediction, confidence, ground_truth
            FROM inference_logs
            {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df.to_dict("records")

    except Exception as e:
        logger.error(f"❌ Failed to fetch logs: {str(e)}")
        return []


def db_get_prediction_counts() -> Dict:
    """
    Return total and labelled prediction counts.
    Used by /stats to populate the summary section.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM inference_logs")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM inference_logs WHERE ground_truth IS NOT NULL")
        labelled = cursor.fetchone()[0]

        conn.close()
        return {"total": total, "labelled": labelled}

    except Exception as e:
        logger.error(f"❌ Failed to get prediction counts: {str(e)}")
        return {"total": 0, "labelled": 0}


def db_log_retraining_event(triggered_by: str, reason: str, training_samples: int) -> int:
    """
    Write a retraining event to the audit log.

    Returns:
        retraining_id — used to update the record with outcome later
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO retraining_history
            (triggered_by, reason, training_samples, status)
            VALUES (?, ?, ?, ?)
        """, (triggered_by, reason, training_samples, "submitted"))

        conn.commit()
        retraining_id = cursor.lastrowid
        conn.close()
        return retraining_id

    except Exception as e:
        logger.error(f"❌ Failed to log retraining event: {str(e)}")
        raise


def db_update_retraining_outcome(retraining_id: int, status: str, new_accuracy: float = None):
    """
    Update a retraining record with its outcome once the pipeline completes.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE retraining_history
            SET status = ?, new_accuracy = ?
            WHERE id = ?
        """, (status, new_accuracy, retraining_id))

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"❌ Failed to update retraining outcome: {str(e)}")


# ============================================================================
# PYDANTIC MODELS (Input / Output Validation)
# ============================================================================

class PatientData(BaseModel):
    """Patient measurement data — 30 features from breast cancer dataset."""
    radius_mean: float = Field(..., description="Mean radius of cells")
    texture_mean: float = Field(..., description="Mean texture of cells")
    perimeter_mean: float = Field(..., description="Mean perimeter of cells")
    area_mean: float = Field(..., description="Mean area of cells")
    smoothness_mean: float = Field(..., description="Mean smoothness of cells")
    compactness_mean: float = Field(..., description="Mean compactness of cells")
    concavity_mean: float = Field(..., description="Mean concavity of cells")
    concave_points_mean: float = Field(..., description="Mean concave points")
    symmetry_mean: float = Field(..., description="Mean symmetry of cells")
    fractal_dimension_mean: float = Field(..., description="Mean fractal dimension")
    radius_se: float = Field(..., description="Radius standard error")
    texture_se: float = Field(..., description="Texture standard error")
    perimeter_se: float = Field(..., description="Perimeter standard error")
    area_se: float = Field(..., description="Area standard error")
    smoothness_se: float = Field(..., description="Smoothness standard error")
    compactness_se: float = Field(..., description="Compactness standard error")
    concavity_se: float = Field(..., description="Concavity standard error")
    concave_points_se: float = Field(..., description="Concave points standard error")
    symmetry_se: float = Field(..., description="Symmetry standard error")
    fractal_dimension_se: float = Field(..., description="Fractal dimension standard error")
    radius_worst: float = Field(..., description="Worst radius")
    texture_worst: float = Field(..., description="Worst texture")
    perimeter_worst: float = Field(..., description="Worst perimeter")
    area_worst: float = Field(..., description="Worst area")
    smoothness_worst: float = Field(..., description="Worst smoothness")
    compactness_worst: float = Field(..., description="Worst compactness")
    concavity_worst: float = Field(..., description="Worst concavity")
    concave_points_worst: float = Field(..., description="Worst concave points")
    symmetry_worst: float = Field(..., description="Worst symmetry")
    fractal_dimension_worst: float = Field(..., description="Worst fractal dimension")


class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""
    success: bool
    prediction_id: int
    diagnosis: str
    diagnosis_code: int
    confidence: float
    probability_no_cancer: float
    probability_cancer: float
    model_version: str
    message: str


class GroundTruthFeedback(BaseModel):
    """Payload for /feedback — submit the real label after a prediction."""
    prediction_id: int = Field(..., description="ID returned from /predict")
    actual_diagnosis: int = Field(..., description="Actual label: 0=no cancer, 1=cancer")


class PerformanceStats(BaseModel):
    """Response from /stats endpoint."""
    total_predictions: int
    labelled_predictions: int
    recent_accuracy: float
    recent_precision: float
    recent_recall: float
    recent_f1_score: float
    drift_detected: bool
    current_model_version: str


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def calculate_performance_metrics() -> Dict:
    """
    Calculate live model performance from labelled predictions in the last 7 days.

    Drift is flagged when accuracy drops below 85% — at which point /stats
    will automatically trigger retraining in the background.

    Returns:
        accuracy, precision, recall, f1_score, drift_detected flag
    """
    try:
        conn = _get_connection()
        query = """
            SELECT prediction, ground_truth
            FROM inference_logs
            WHERE ground_truth IS NOT NULL
            AND timestamp > datetime('now', '-7 days')
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if len(df) == 0:
            logger.warning("⚠️ No labelled data in the last 7 days")
            return {
                "total_predictions": 0,
                "labelled_predictions": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "drift_detected": False
            }

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy  = accuracy_score(df['ground_truth'], df['prediction'])
        precision = precision_score(df['ground_truth'], df['prediction'], average='weighted', zero_division=0)
        recall    = recall_score(df['ground_truth'], df['prediction'], average='weighted', zero_division=0)
        f1        = f1_score(df['ground_truth'], df['prediction'], average='weighted', zero_division=0)

        # Drift threshold — accuracy below 85% means the model needs retraining
        drift_detected = accuracy < 0.85

        metrics = {
            "total_predictions":    len(df),
            "labelled_predictions": len(df),
            "accuracy":             accuracy,
            "precision":            precision,
            "recall":               recall,
            "f1_score":             f1,
            "drift_detected":       drift_detected
        }

        logger.info(f"📊 Metrics — Accuracy: {accuracy:.2%}, Drift: {drift_detected}")

        # Log monitoring snapshot to MLflow for experiment tracking
        mlflow.set_experiment("monitoring")
        with mlflow.start_run(run_name="performance_metrics"):
            mlflow.log_metric("accuracy",        accuracy)
            mlflow.log_metric("precision",       precision)
            mlflow.log_metric("recall",          recall)
            mlflow.log_metric("f1_score",        f1)
            mlflow.log_metric("drift_detected",  int(drift_detected))

        return metrics

    except Exception as e:
        logger.error(f"❌ Failed to calculate metrics: {str(e)}")
        return {}


# ============================================================================
# MODEL LOADING (ONCE AT STARTUP)
# ============================================================================

def load_model() -> bool:
    global _model, _preprocess_pipeline

    try:
        logger.info("🔄 Loading production model from ZenML...")

        model_version = client.get_model_version(
            "breast_cancer_classifier",
            ModelStages.PRODUCTION  # positional — not a keyword argument
        )
        logger.info(f"✅ Found production model version: {model_version.name}")

        _model = model_version.get_artifact("sklearn_classifier").load()
        logger.info("✅ Loaded sklearn classifier")

        _preprocess_pipeline = model_version.get_artifact("preprocess_pipeline").load()
        logger.info("✅ Loaded preprocessing pipeline")

        return True

    except Exception as e:
        logger.error(f"❌ ZenML model load failed: {str(e)}")

        try:
            logger.info("⚠️ Falling back to MLflow...")
            mlflow_client = mlflow.tracking.MlflowClient("http://localhost:5000")

            # sklearn_classifier — the artifact name, not the ZenML model name
            latest_versions = mlflow_client.get_latest_versions("sklearn_classifier")

            if latest_versions:
                latest    = latest_versions[0]
                model_uri = f"runs:/{latest.run_id}/sklearn_classifier"
                _model    = mlflow.sklearn.load_model(model_uri)
                logger.info(f"✅ Loaded from MLflow fallback: run {latest.run_id}")
                return True

        except Exception as fallback_error:
            logger.error(f"❌ MLflow fallback also failed: {str(fallback_error)}")

        return False

# ============================================================================
# RETRAINING PIPELINE (RUNS IN BACKGROUND THREAD)
# ============================================================================

def run_retraining_pipeline(labelled_records: pd.DataFrame, retraining_id: int):
    """
    Execute the ZenML training pipeline with accumulated labelled data.

    Runs in a background thread via ThreadPoolExecutor so the API stays
    responsive while retraining is in progress.

    After the pipeline completes:
    1. Updates the retraining audit log with outcome
    2. Calls load_model() to hot-swap the new production model into memory
    """
    try:
        logger.info(f"🔄 Retraining pipeline starting with {len(labelled_records)} samples...")

        from pipelines import training

        training.with_options(
            config_path="configs/training_rf.yaml"
        )(new_data=labelled_records)

        logger.info("✅ Retraining pipeline completed successfully")

        # Update audit log with success
        db_update_retraining_outcome(
            retraining_id=retraining_id,
            status="completed"
        )

        # Hot-swap the model — reload production model into memory without restarting
        logger.info("🔄 Reloading model after retraining...")
        success = load_model()

        if success:
            logger.info("✅ New model loaded — API now serving updated model")
        else:
            logger.error("❌ Model reload failed after retraining")

    except Exception as e:
        logger.error(f"❌ Retraining pipeline failed: {str(e)}")
        db_update_retraining_outcome(
            retraining_id=retraining_id,
            status="failed"
        )


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and load model when the API starts."""
    logger.info("🚀 Starting Breast Cancer Detector API v2.1...")
    init_database()
    success = load_model()
    if not success:
        logger.warning("⚠️ Model not loaded at startup. /predict will fail until model is available.")


# ============================================================================
# INFO ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API overview and available endpoints."""
    return {
        "name":        "Breast Cancer Detector API",
        "version":     "2.1.0",
        "description": "Production inference with continuous learning feedback loop",
        "endpoints": {
            "health":       "/health",
            "predict":      "/predict",
            "predict_batch": "/predict-batch",
            "feedback":     "/feedback",
            "retrain":      "/retrain",
            "reload_model": "/reload-model",
            "stats":        "/stats",
            "logs":         "/logs",
            "docs":         "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Check whether the API and model are ready to serve predictions."""
    model_loaded = _model is not None and _preprocess_pipeline is not None

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — check startup logs")

    return {
        "status":           "healthy",
        "model_loaded":     True,
        "model_name":       "sklearn_classifier",
        "model_version":    "production",
        "database_exists":  Path(_db_path).exists()
    }


@app.get("/model-info", tags=["Info"])
async def model_info():
    """Return details about the currently loaded model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type":    type(_model).__name__,
        "model_params":  _model.get_params(),
        "model_version": "production",
        "pipeline_steps": [step[0] for step in _preprocess_pipeline.steps] if _preprocess_pipeline else []
    }


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(patient: PatientData):
    """
    Make a single prediction.

    Flow:
        1. Receive patient features
        2. Apply preprocessing pipeline (same as training)
        3. Run model inference
        4. Log prediction to database (ground_truth=NULL)
        5. Return prediction + prediction_id to caller

    The prediction_id should be stored by the caller — they'll need it
    to submit ground truth later via /feedback.
    """
    if _model is None or _preprocess_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Convert to DataFrame — add dummy target column for pipeline compatibility
        patient_dict = patient.dict()
        patient_df   = pd.DataFrame([patient_dict])
        patient_df['target'] = 1

        # Apply the same preprocessing used during training
        patient_processed = _preprocess_pipeline.transform(patient_df)
        patient_processed = patient_processed.drop(columns=['target'], errors='ignore')

        # Run inference
        prediction = _model.predict(patient_processed)[0]

        # Get class probabilities if the model supports it
        try:
            probabilities     = _model.predict_proba(patient_processed)[0]
            prob_no_cancer    = float(probabilities[0])
            prob_cancer       = float(probabilities[1])
        except AttributeError:
            # Model doesn't support predict_proba (e.g. LinearSVC)
            prob_no_cancer = 0.0
            prob_cancer    = 0.0

        confidence       = max(prob_no_cancer, prob_cancer)
        diagnosis        = "cancer" if prediction == 1 else "no_cancer"
        response_time_ms = (time.time() - start_time) * 1000

        # Log to database — ground_truth is NULL until /feedback is called
        prediction_id = db_log_prediction(
            input_features        = patient_dict,
            prediction            = int(prediction),
            confidence            = confidence,
            probability_no_cancer = prob_no_cancer,
            probability_cancer    = prob_cancer,
            model_version         = "production",
            response_time_ms      = response_time_ms
        )

        # Log inference event to MLflow for experiment tracking
        mlflow.set_experiment("inference")
        with mlflow.start_run(run_name="api_prediction"):
            mlflow.log_metric("prediction_confidence", confidence)
            mlflow.log_metric("response_time_ms",      response_time_ms)
            mlflow.log_param("diagnosis",              diagnosis)

        return PredictionResponse(
            success               = True,
            prediction_id         = prediction_id,
            diagnosis             = diagnosis,
            diagnosis_code        = int(prediction),
            confidence            = confidence,
            probability_no_cancer = prob_no_cancer,
            probability_cancer    = prob_cancer,
            model_version         = "production",
            message               = f"{diagnosis} ({confidence:.1%}). Use prediction_id {prediction_id} to submit feedback."
        )

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", tags=["Inference"])
async def predict_batch(patients: List[PatientData]):
    """
    Make predictions for multiple patients in a single request.

    Each prediction is logged individually so feedback can be submitted
    per patient using their individual prediction_id.
    """
    if _model is None or _preprocess_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        patients_data = [p.dict() for p in patients]
        patients_df   = pd.DataFrame(patients_data)
        patients_df['target'] = 1

        patients_processed = _preprocess_pipeline.transform(patients_df)
        patients_processed = patients_processed.drop(columns=['target'], errors='ignore')

        predictions = _model.predict(patients_processed)

        try:
            probabilities = _model.predict_proba(patients_processed)
        except AttributeError:
            probabilities = None

        predictions_list = []
        for idx, (patient, pred) in enumerate(zip(patients, predictions)):
            diagnosis = "cancer" if pred == 1 else "no_cancer"

            if probabilities is not None:
                prob_no_cancer = float(probabilities[idx][0])
                prob_cancer    = float(probabilities[idx][1])
                confidence     = max(prob_no_cancer, prob_cancer)
            else:
                prob_no_cancer = 0.0
                prob_cancer    = 0.0
                confidence     = 0.5

            prediction_id = db_log_prediction(
                input_features        = patient.dict(),
                prediction            = int(pred),
                confidence            = confidence,
                probability_no_cancer = prob_no_cancer,
                probability_cancer    = prob_cancer,
                model_version         = "production",
                response_time_ms      = 0
            )

            predictions_list.append({
                "prediction_id":  prediction_id,
                "patient_index":  idx,
                "diagnosis":      diagnosis,
                "confidence":     confidence
            })

        mlflow.set_experiment("inference")
        with mlflow.start_run(run_name="api_batch_prediction"):
            mlflow.log_param("batch_size", len(patients))

        return {
            "success":     True,
            "batch_size":  len(patients),
            "predictions": predictions_list
        }

    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONTINUOUS LEARNING: FEEDBACK
# ============================================================================

@app.post("/feedback", tags=["Continuous Learning"])
async def submit_feedback(feedback: GroundTruthFeedback):
    """
    Submit the real label for a past prediction.

    This is the entry point for the continuous learning loop:
        1. /predict returns a prediction_id
        2. Reality reveals the actual label (biopsy result, analyst review, etc.)
        3. Caller submits prediction_id + actual label here
        4. Database record is updated with ground_truth
        5. Labelled data accumulates over time
        6. /stats uses this data to detect drift
        7. Drift → automatic retraining

    Without this endpoint, the model can never improve from production data.
    """
    try:
        if feedback.actual_diagnosis not in [0, 1]:
            raise HTTPException(status_code=400, detail="actual_diagnosis must be 0 (no cancer) or 1 (cancer)")

        success = db_update_ground_truth(
            prediction_id    = feedback.prediction_id,
            actual_diagnosis = feedback.actual_diagnosis
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update ground truth in database")

        # Count total labelled records after this update
        labelled_records = db_fetch_labelled_records()

        # Log feedback event to MLflow
        mlflow.set_experiment("monitoring")
        with mlflow.start_run(run_name="feedback_received"):
            mlflow.log_param("prediction_id",  feedback.prediction_id)
            mlflow.log_param("ground_truth",   feedback.actual_diagnosis)
            mlflow.log_metric("total_labelled", len(labelled_records))

        logger.info(f"✅ Feedback recorded. Total labelled samples: {len(labelled_records)}")

        return {
            "success":             True,
            "prediction_id":       feedback.prediction_id,
            "ground_truth":        feedback.actual_diagnosis,
            "total_labelled_data": len(labelled_records),
            "message":             f"Ground truth recorded. {len(labelled_records)} labelled samples now available for retraining."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MONITORING & STATS
# ============================================================================

@app.get("/stats", response_model=PerformanceStats, tags=["Monitoring"])
async def get_stats():
    """
    Return current model performance metrics calculated from labelled predictions.

    Also auto-triggers retraining in the background if drift is detected —
    so simply calling /stats is enough to keep the model healthy automatically.

    Drift = accuracy drops below 85% over the last 7 days of labelled data.
    """
    try:
        metrics = calculate_performance_metrics()
        counts  = db_get_prediction_counts()

        # Auto-trigger retraining if drift detected — runs in background
        if metrics.get("drift_detected"):
            logger.warning("⚠️ Drift detected in /stats — triggering automatic retraining")
            labelled_records = db_fetch_labelled_records()

            if len(labelled_records) >= 50:
                retraining_id = db_log_retraining_event(
                    triggered_by     = "auto_drift_detection",
                    reason           = f"Accuracy dropped to {metrics.get('accuracy', 0):.2%} — below 85% threshold",
                    training_samples = len(labelled_records)
                )
                loop = asyncio.get_event_loop()
                loop.run_in_executor(
                    executor,
                    run_retraining_pipeline,
                    labelled_records,
                    retraining_id
                )
                logger.info("🔄 Retraining started in background thread")
            else:
                logger.warning(f"⚠️ Drift detected but only {len(labelled_records)} labelled samples — need 50 to retrain")

        return PerformanceStats(
            total_predictions     = counts["total"],
            labelled_predictions  = counts["labelled"],
            recent_accuracy       = metrics.get("accuracy",  0) or 0,
            recent_precision      = metrics.get("precision", 0) or 0,
            recent_recall         = metrics.get("recall",    0) or 0,
            recent_f1_score       = metrics.get("f1_score",  0) or 0,
            drift_detected        = metrics.get("drift_detected", False),
            current_model_version = "production"
        )

    except Exception as e:
        logger.error(f"❌ Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs", tags=["Monitoring"])
async def get_logs(limit: int = 20, labelled_only: bool = False):
    """
    View recent prediction records from the database.

    Args:
        limit:         Number of records to return (default 20)
        labelled_only: If True, only show predictions with ground truth
    """
    try:
        return db_fetch_recent_logs(limit=limit, labelled_only=labelled_only)
    except Exception as e:
        logger.error(f"❌ Logs error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RETRAINING & MODEL RELOAD
# ============================================================================

@app.post("/retrain", tags=["Continuous Learning"])
async def trigger_retrain(force: bool = False):
    """
    Manually trigger retraining with all accumulated labelled data.

    Retraining runs in a background thread — the API continues serving
    predictions while the new model trains.

    Workflow:
        1. Fetch all labelled records from database
        2. Check minimum threshold (50 samples) — bypass with force=True
        3. Log retraining event to audit table
        4. Launch ZenML training pipeline in background thread
        5. Pipeline promotes best model to production in ZenML
        6. load_model() hot-swaps the new model into memory
        7. API now serves the new model — zero downtime

    Args:
        force: Skip the minimum sample threshold check
    """
    try:
        labelled_records = db_fetch_labelled_records()

        if len(labelled_records) == 0:
            raise HTTPException(status_code=400, detail="No labelled data available for retraining")

        if len(labelled_records) < 50 and not force:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(labelled_records)} labelled samples. Need at least 50, or pass force=True to override."
            )

        logger.info(f"🔄 Manual retraining triggered with {len(labelled_records)} samples")

        # Write to audit log before starting
        retraining_id = db_log_retraining_event(
            triggered_by     = "manual_api_request",
            reason           = f"Manual trigger via /retrain with {len(labelled_records)} labelled samples",
            training_samples = len(labelled_records)
        )

        # Launch pipeline in background — API stays responsive
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            executor,
            run_retraining_pipeline,
            labelled_records,
            retraining_id
        )

        return {
            "success":         True,
            "retraining_id":   retraining_id,
            "labelled_samples": len(labelled_records),
            "status":          "retraining started in background",
            "message":         "Pipeline is running. Call /stats or /health to monitor progress. Model will auto-reload when complete."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Retrain error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model", tags=["Continuous Learning"])
async def reload_model():
    """
    Hot-swap the in-memory model with the current production model from ZenML.

    Called automatically after retraining completes — but can also be called
    manually if you've promoted a new model version in ZenML and want the
    API to pick it up immediately without restarting.

    Zero downtime — the old model keeps serving until the new one is loaded.
    """
    try:
        logger.info("🔄 Manual model reload requested")
        success = load_model()

        if success:
            logger.info("✅ Model reloaded successfully")
            return {
                "success": True,
                "message": "Production model reloaded into memory. API is now serving the updated model."
            }
        else:
            raise HTTPException(status_code=500, detail="Model reload failed — check logs for details")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")