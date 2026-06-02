"""
Enterprise Financial Fraud Inference API
=============================================================

FastAPI service designed for hyper-scale production environments.
Features:
- Asynchronous MySQL Connection Pooling (Fail-Fast initialized)
- Atomic Version-Pointer Model Hot-Swapping (Zero-Downtime, Thread-Safe)
- Strict Feature Schema Enforcements (Guards against upstream drift)
- Event-Driven Job Queues for Retraining (Decoupled execution)
- Structured JSON Logging (Datadog/Elasticsearch compatible)
- In-memory rate limiting & Inference Timeouts
- .env configuration for 12-factor app deployment
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import aiomysql
import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from zenml.client import Client
from zenml.enums import ModelStages

# ============================================================================
# CONFIGURATION & ENVIRONMENT LOAD
# ============================================================================
load_dotenv()  # Load 12-factor secrets from .env

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mysqlrootpassword")
MYSQL_DB = os.getenv("MYSQL_DB", "fraud_inference_db")

API_KEY = os.getenv("API_KEY", "super-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Define the exact physical order of columns required by the scikit-learn model
EXPECTED_FEATURE_ORDER = [
    "age", "gender", "category", "amount", 
    "count_1_day", "count_7_days", "count_30_days"
]

# ============================================================================
# STRUCTURED JSON LOGGING
# ============================================================================
# Replace basic unstructured text logs with strictly formatted JSON logs
logger = logging.getLogger("inference_api")
logger.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

# Rate Limiter Setup
limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# ATOMIC MODEL LOADING & STATE MANAGEMENT
# ============================================================================
def fetch_production_model_from_zenml() -> Tuple[str, object, object]:
    """
    Connects to ZenML to fetch the active production model.
    Returns: (version_name, model_artifact, preprocessor_artifact)
    """
    logger.info("Connecting to ZenML artifact store...")
    client = Client()
    model_version = client.get_model_version("fraud_detection_model", ModelStages.PRODUCTION)
    
    # Load physical artifacts into memory
    model = model_version.get_artifact("fraud_trained_model").load()
    preprocessor = model_version.get_artifact("fraud_preprocessing_pipeline").load()
    
    version_id = model_version.name
    logger.info("Successfully fetched model artifacts", extra={"model_version": version_id})
    return version_id, model, preprocessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan event. Initializes critical infrastructure.
    If database or initial model loading fails, the app FAILS FAST and crashes,
    ensuring a load balancer won't route traffic to a broken node.
    """
    logger.info("Starting up FastAPI Inference Server...")
    
    # 1. Initialize DB (Fails fast if down)
    app.state.db_pool = await init_database_pool()
    
    # 2. Initialize Model Version Pointer Dictionary and Lock
    app.state.models = {}
    app.state.model_reload_lock = asyncio.Lock()
    
    # 3. Load initial model and set active pointer
    version_id, model, preprocessor = fetch_production_model_from_zenml()
    app.state.models[version_id] = (model, preprocessor)
    app.state.active_model_version = version_id
    
    logger.info("Startup complete. Ready for traffic.", extra={"active_model": version_id})
    
    yield  # Yield control back to FastAPI to begin serving requests
    
    # Shutdown logic
    if app.state.db_pool:
        app.state.db_pool.close()
        await app.state.db_pool.wait_closed()


app = FastAPI(
    title="Financial Fraud Detector API",
    description="Enterprise inference API with atomic model swapping and JSON logging.",
    version="2.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Dependencies ---
def get_active_model(request: Request) -> Tuple[str, object, object]:
    """Dependency injection to resolve the active model atomically per request."""
    active_id = request.app.state.active_model_version
    model, preprocessor = request.app.state.models.get(active_id, (None, None))
    if model is None:
        raise HTTPException(status_code=503, detail="Active model is not loaded.")
    return active_id, model, preprocessor

def get_db_pool(request: Request) -> aiomysql.Pool:
    """Dependency injection to access the persistent MySQL pool."""
    if request.app.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database pool is unavailable.")
    return request.app.state.db_pool

def verify_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the X-API-Key header for sensitive ops routes."""
    if api_key_header != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key_header


# ============================================================================
# ASYNC DATABASE LAYER (aiomysql)
# ============================================================================
async def init_database_pool() -> aiomysql.Pool:
    """
    Establishes the MySQL connection pool and creates tables if they don't exist.
    WARNING: We intentionally do not suppress exceptions here. If the DB is down,
    we want the server to violently crash (Fail-Fast pattern).
    """
    logger.info("Initializing MySQL connection pool...")
    pool = await aiomysql.create_pool(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, db=MYSQL_DB,
        autocommit=True, minsize=2, maxsize=20
    )
    
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            # Table for real-time inference logging
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id                      INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp               DATETIME DEFAULT CURRENT_TIMESTAMP,
                    input_features          TEXT NOT NULL,
                    prediction              INT NOT NULL,
                    confidence              DOUBLE NOT NULL,
                    probability_legit       DOUBLE,
                    probability_fraud       DOUBLE,
                    ground_truth            INT,
                    ground_truth_timestamp  DATETIME,
                    model_version           VARCHAR(255),
                    api_response_time_ms    DOUBLE
                )
            """)
            # Table for event-driven decoupled training jobs
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS retraining_jobs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    started_at DATETIME NULL,
                    completed_at DATETIME NULL,
                    error_message TEXT NULL
                )
            """)
    logger.info("Database schemas verified and connection pool ready.")
    return pool


async def db_log_prediction(
    pool: aiomysql.Pool,
    input_features: Dict,
    prediction: int,
    confidence: float,
    probability_legit: float,
    probability_fraud: float,
    model_version: str,
    response_time_ms: float,
):
    """Background task to insert predictions asynchronously without blocking the API."""
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO inference_logs
                    (input_features, prediction, confidence, probability_legit,
                     probability_fraud, model_version, api_response_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        json.dumps(input_features), prediction, confidence,
                        probability_legit, probability_fraud, model_version, response_time_ms
                    )
                )
    except Exception as e:
        logger.error("Failed to write inference log to DB", extra={"error": str(e)})


async def db_update_ground_truth(pool: aiomysql.Pool, prediction_id: int, actual_fraud: int) -> bool:
    """Updates a previous prediction with true labels for continuous learning."""
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE inference_logs SET ground_truth = %s, ground_truth_timestamp = CURRENT_TIMESTAMP WHERE id = %s",
                    (actual_fraud, prediction_id),
                )
        return True
    except Exception as e:
        logger.error("Failed to update ground truth", extra={"error": str(e), "prediction_id": prediction_id})
        return False


# ============================================================================
# PYDANTIC MODELS & VALIDATION
# ============================================================================
class TransactionData(BaseModel):
    age: str = Field(..., description="Age group (e.g. '1', '2', '3', 'U')")
    gender: str = Field(..., description="Gender (e.g. 'M', 'F', 'E')")
    category: str = Field(..., description="Transaction category")
    amount: float = Field(..., description="Transaction amount")
    count_1_day: float = Field(..., description="Transactions in last 1 day")
    count_7_days: float = Field(..., description="Transactions in last 7 days")
    count_30_days: float = Field(..., description="Transactions in last 30 days")


class PredictionResponse(BaseModel):
    success: bool
    diagnosis: str
    diagnosis_code: int
    confidence: float
    probability_legit: float
    probability_fraud: float
    message: str


class FeedbackData(BaseModel):
    prediction_id: int = Field(..., description="ID returned from logs endpoint")
    actual_fraud: int = Field(..., description="Actual label: 0=legit, 1=fraud")


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

def _predict_logic(model, preprocessor, df: pd.DataFrame):
    """
    Isolated CPU-bound processing logic. 
    Converts data types, runs the scikit-learn pipeline, and extracts probabilities.
    """
    for col in ["age", "gender", "category"]:
        df[col] = df[col].astype(str).str.replace(r"[\"']", "", regex=True)

    processed_df = preprocessor.transform(df)
    predictions = model.predict(processed_df)
    
    try:
        probs = model.predict_proba(processed_df)
    except AttributeError:
        probs = [[1.0, 0.0] if p == 0 else [0.0, 1.0] for p in predictions]
        
    return predictions, probs


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@limiter.limit("60/minute")
async def predict(
    request: Request,
    transaction: TransactionData,
    background_tasks: BackgroundTasks,
    model_bundle: Tuple[str, object, object] = Depends(get_active_model),
    db_pool: aiomysql.Pool = Depends(get_db_pool)
):
    """
    Real-time fraud prediction endpoint.
    Applies strict feature ordering, enforces a 2-second timeout, and uses JSON logs.
    """
    version_id, model, preprocessor = model_bundle
    start_time = time.time()
    data_dict = transaction.dict()

    # STRICT SCHEMA VALIDATION: Force Pandas to construct columns in EXACT expected physical order
    # This prevents silent pipeline corruption if upstream JSON orders shift.
    try:
        ordered_data = {col: [data_dict[col]] for col in EXPECTED_FEATURE_ORDER}
        df = pd.DataFrame(ordered_data)
    except KeyError as e:
        logger.error("Missing required feature", extra={"missing_feature": str(e)})
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    try:
        # Prevent runaway inference computations
        predictions, probs = await asyncio.wait_for(
            asyncio.to_thread(_predict_logic, model, preprocessor, df),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        logger.error("Inference Timeout", extra={"model_version": version_id})
        raise HTTPException(status_code=504, detail="Prediction timeout exceeded.")

    prediction = predictions[0]
    prob_legit, prob_fraud = float(probs[0][0]), float(probs[0][1])
    confidence = max(prob_legit, prob_fraud)
    diagnosis = "Fraudulent" if prediction == 1 else "Legitimate"
    response_time_ms = (time.time() - start_time) * 1000

    # Structured Logging
    logger.info("Prediction successful", extra={
        "event": "predict",
        "latency_ms": round(response_time_ms, 2),
        "model_version": version_id,
        "prediction": int(prediction),
        "confidence": confidence
    })

    background_tasks.add_task(
        db_log_prediction, db_pool, data_dict, int(prediction),
        confidence, prob_legit, prob_fraud, version_id, response_time_ms
    )

    return PredictionResponse(
        success=True, diagnosis=diagnosis, diagnosis_code=int(prediction),
        confidence=confidence, probability_legit=prob_legit, probability_fraud=prob_fraud,
        message=f"{diagnosis} ({confidence:.1%}).",
    )


# ============================================================================
# EVENT-DRIVEN OPS ENDPOINTS
# ============================================================================

@app.post("/retrain", tags=["Ops"], dependencies=[Depends(verify_api_key)])
async def trigger_retraining(request: Request, db_pool: aiomysql.Pool = Depends(get_db_pool)):
    """
    Decoupled Retraining Trigger.
    Instead of executing heavy Python scripts that block the API CPU, this simply 
    inserts a 'pending' job into MySQL. A separate worker process (training_worker.py)
    will pick it up and do the heavy lifting elsewhere.
    """
    try:
        async with db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO retraining_jobs (status) VALUES ('pending')")
        logger.info("Retraining job queued successfully.", extra={"event": "queue_retraining"})
        return {"message": "Retraining job queued! A background worker will execute it."}
    except Exception as e:
        logger.error("Failed to queue retraining job", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Database queue failed.")


@app.post("/reload-model", tags=["Ops"], dependencies=[Depends(verify_api_key)])
async def reload_model_hot_swap(request: Request):
    """
    Atomic Model Hot-Swapping with Concurrency Protection.
    Safely loads the new model into a memory dictionary BEFORE modifying the active pointer.
    Uses an asyncio.Lock to prevent multiple admins from triggering collisions simultaneously.
    """
    logger.info("Initiating atomic model hot-swap sequence.")
    
    # Protect against concurrent reload requests
    async with request.app.state.model_reload_lock:
        try:
            # Load the new artifacts from ZenML (takes time)
            new_version_id, new_model, new_preprocessor = fetch_production_model_from_zenml()
            
            # Store in the dictionary memory pool
            request.app.state.models[new_version_id] = (new_model, new_preprocessor)
            
            # Atomically swap the active pointer in 1 millisecond
            old_version_id = request.app.state.active_model_version
            request.app.state.active_model_version = new_version_id
            
            logger.info("Model hot-swap complete.", extra={
                "event": "model_reload",
                "old_version": old_version_id,
                "new_version": new_version_id
            })
            
            # Optional: Delete the old model from dictionary to free RAM
            if old_version_id != new_version_id and old_version_id in request.app.state.models:
                del request.app.state.models[old_version_id]
                
            return {"message": f"Successfully hot-swapped to {new_version_id}"}
        except Exception as e:
            logger.error("Hot-swap failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")
