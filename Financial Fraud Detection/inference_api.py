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
import redis.asyncio as redis_async
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse, RedirectResponse
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

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

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
def fetch_model_from_zenml(stage: ModelStages = ModelStages.PRODUCTION) -> Tuple[str, object, object]:
    """
    Connects to ZenML to fetch the active model for a given stage.
    Returns: (version_name, model_artifact, preprocessor_artifact)
    """
    logger.info("Connecting to ZenML artifact store...")
    client = Client()
    model_version = client.get_model_version("fraud_detection_model", stage)
    
    # Load physical artifacts into memory
    model = model_version.get_artifact("fraud_trained_model").load()
    preprocessor = model_version.get_artifact("fraud_preprocess_pipeline").load()
    
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
    
    # 1.5 Initialize Redis Feature Store
    logger.info("Connecting to Redis Feature Store...")
    app.state.redis_pool = redis_async.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    
    # 2. Initialize Model Version Pointer Dictionary and Lock
    app.state.models = {}
    app.state.model_reload_lock = asyncio.Lock()
    app.state.shadow_model_version = None
    
    # 3. Load initial model and set active pointer
    version_id, model, preprocessor = fetch_model_from_zenml(ModelStages.PRODUCTION)
    app.state.models[version_id] = (model, preprocessor)
    app.state.active_model_version = version_id
    
    logger.info("Startup complete. Ready for traffic.", extra={"active_model": version_id})
    
    yield  # Yield control back to FastAPI to begin serving requests
    
    # Shutdown logic
    if app.state.db_pool:
        app.state.db_pool.close()
        await app.state.db_pool.wait_closed()
    if getattr(app.state, "redis_pool", None):
        await app.state.redis_pool.close()


app = FastAPI(
    title="Financial Fraud Detector API",
    description="Enterprise inference API with atomic model swapping and JSON logging.",
    version="2.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Advanced Observability: Mount Prometheus Metrics
Instrumentator().instrument(app).expose(app)


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

def get_redis_pool(request: Request) -> redis_async.Redis:
    """Dependency injection to access Redis Feature Store."""
    if not getattr(request.app.state, "redis_pool", None):
        raise HTTPException(status_code=503, detail="Redis pool is unavailable.")
    return request.app.state.redis_pool

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
                    shadow_prediction       INT,
                    api_response_time_ms    DOUBLE
                )
            """)
            # Ensure any newer columns exist if the table was pre-existing
            for col_name, col_type in [
                ("probability_legit", "DOUBLE"),
                ("probability_fraud", "DOUBLE"),
                ("shadow_prediction", "INT"),
                ("api_response_time_ms", "DOUBLE")
            ]:
                try:
                    await cur.execute(f"ALTER TABLE inference_logs ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Database migration: added missing column '{col_name}' to inference_logs.")
                except Exception:
                    # Ignore error if the column already exists
                    pass

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
    shadow_prediction: int = None
):
    """Background task to insert predictions asynchronously without blocking the API."""
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO inference_logs
                    (input_features, prediction, confidence, probability_legit,
                     probability_fraud, model_version, shadow_prediction, api_response_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        json.dumps(input_features), prediction, confidence,
                        probability_legit, probability_fraud, model_version, shadow_prediction, response_time_ms
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
    customer_id: str = Field(..., description="Unique customer identifier")
    age: str = Field(..., description="Age group (e.g. '1', '2', '3', 'U')")
    gender: str = Field(..., description="Gender (e.g. 'M', 'F', 'E')")
    category: str = Field(..., description="Transaction category")
    amount: float = Field(..., description="Transaction amount")


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
# ROOT & UTILITY ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Serve a premium, informative landing page and API status dashboard."""
    active_version = getattr(request.app.state, "active_model_version", "Unknown")
    shadow_version = getattr(request.app.state, "shadow_model_version", "None")
    if shadow_version is None:
        shadow_version = "None"
    loaded_versions = list(getattr(request.app.state, "models", {}).keys())
    db_pool = getattr(request.app.state, "db_pool", None)
    
    db_status = "Online" if db_pool is not None else "Offline"
    db_color = "#10b981" if db_pool is not None else "#ef4444"
    
    template_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Hydrate dynamic variables
        html_content = html_content.replace("{{active_version}}", active_version)
        html_content = html_content.replace("{{shadow_version}}", shadow_version)
        html_content = html_content.replace("{{db_status}}", db_status)
        html_content = html_content.replace("{{db_color}}", db_color)
        html_content = html_content.replace(
            "{{loaded_versions}}", 
            ", ".join(loaded_versions) if loaded_versions else "None"
        )
    except Exception as e:
        logger.error("Failed to load dashboard HTML template", extra={"error": str(e)})
        return HTMLResponse(content=f"<h1>Error</h1><p>Failed to load dashboard template: {e}</p>", status_code=500)
        
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Gracefully handle browser favicon requests."""
    return {"message": "No favicon"}


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
    db_pool: aiomysql.Pool = Depends(get_db_pool),
    redis_pool = Depends(get_redis_pool)
):
    """
    Real-time fraud prediction endpoint.
    Applies strict feature ordering, enforces a 2-second timeout, and uses JSON logs.
    """
    version_id, model, preprocessor = model_bundle
    start_time = time.time()
    data_dict = transaction.dict()
    customer_id = data_dict.pop("customer_id")

    # Online Feature Store: Fetch rolling counts
    try:
        redis_key = f"customer:{customer_id}"
        features = await redis_pool.hgetall(redis_key)
        data_dict["count_1_day"] = float(features.get("count_1_day", 0.0))
        data_dict["count_7_days"] = float(features.get("count_7_days", 0.0))
        data_dict["count_30_days"] = float(features.get("count_30_days", 0.0))
        
        # Fire-and-forget: increment counts
        async def increment_redis():
            await redis_pool.hincrbyfloat(redis_key, "count_1_day", 1.0)
            await redis_pool.hincrbyfloat(redis_key, "count_7_days", 1.0)
            await redis_pool.hincrbyfloat(redis_key, "count_30_days", 1.0)
            await redis_pool.expire(redis_key, 30 * 24 * 3600)
        background_tasks.add_task(increment_redis)
    except Exception as e:
        logger.error("Redis Feature Store fetch failed", extra={"error": str(e)})
        # Default to 0.0 if Redis fails
        data_dict["count_1_day"], data_dict["count_7_days"], data_dict["count_30_days"] = 0.0, 0.0, 0.0

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

    # Shadow Model Evaluation (A/B Serving)
    shadow_version_id = request.app.state.shadow_model_version
    shadow_prediction_val = None
    if shadow_version_id:
        shadow_model, shadow_preprocessor = request.app.state.models.get(shadow_version_id, (None, None))
        if shadow_model:
            try:
                shadow_preds, _ = await asyncio.wait_for(
                    asyncio.to_thread(_predict_logic, shadow_model, shadow_preprocessor, df),
                    timeout=1.0
                )
                shadow_prediction_val = int(shadow_preds[0])
            except Exception as e:
                logger.warning("Shadow model prediction failed", extra={"error": str(e)})

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
        confidence, prob_legit, prob_fraud, version_id, response_time_ms, shadow_prediction_val
    )

    return PredictionResponse(
        success=True, diagnosis=diagnosis, diagnosis_code=int(prediction),
        confidence=confidence, probability_legit=prob_legit, probability_fraud=prob_fraud,
        message=f"{diagnosis} ({confidence:.1%}).",
    )


# ============================================================================
# EVENT-DRIVEN OPS ENDPOINTS
# ============================================================================

@app.post("/feedback", tags=["Ops"])
async def receive_feedback(feedback: FeedbackData, db_pool: aiomysql.Pool = Depends(get_db_pool)):
    """Receives ground truth labels and updates the inference logs for continuous learning."""
    success = await db_update_ground_truth(db_pool, feedback.prediction_id, feedback.actual_fraud)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update feedback loop.")
    return {"message": "Feedback successfully recorded."}

@app.post("/retrain", tags=["Ops"], dependencies=[Depends(verify_api_key)])
async def trigger_retraining(request: Request, db_pool: aiomysql.Pool = Depends(get_db_pool)):
    """
    Decoupled Retraining Trigger via Celery Task Queue.
    """
    try:
        from celery_worker import run_training_pipeline_task
        run_training_pipeline_task.delay()
        logger.info("Retraining job dispatched to Celery.", extra={"event": "queue_retraining"})
        return {"message": "Retraining job queued! Celery workers will execute it."}
    except Exception as e:
        logger.error("Failed to dispatch Celery task", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Task queue failed.")


@app.post("/reload-model", tags=["Ops"], dependencies=[Depends(verify_api_key)])
async def reload_model_hot_swap(request: Request, load_as_shadow: bool = False):
    """
    Atomic Model Hot-Swapping with Concurrency Protection.
    Safely loads the new model into a memory dictionary BEFORE modifying the active pointer.
    If load_as_shadow is True, it loads into the shadow pointer.
    Uses an asyncio.Lock to prevent multiple admins from triggering collisions simultaneously.
    """
    logger.info("Initiating atomic model hot-swap sequence.", extra={"load_as_shadow": load_as_shadow})
    
    # Protect against concurrent reload requests
    async with request.app.state.model_reload_lock:
        try:
            # Load the new artifacts from ZenML (takes time)
            if load_as_shadow:
                new_version_id, new_model, new_preprocessor = fetch_model_from_zenml(ModelStages.STAGING)
            else:
                new_version_id, new_model, new_preprocessor = fetch_model_from_zenml(ModelStages.PRODUCTION)
            
            # Store in the dictionary memory pool
            request.app.state.models[new_version_id] = (new_model, new_preprocessor)
            
            if load_as_shadow:
                request.app.state.shadow_model_version = new_version_id
                logger.info("Shadow model loaded.", extra={"new_version": new_version_id})
                return {"message": f"Successfully loaded {new_version_id} as shadow model."}
            else:
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
                    if request.app.state.shadow_model_version != old_version_id:
                        del request.app.state.models[old_version_id]
                    
                return {"message": f"Successfully hot-swapped to {new_version_id}"}
        except Exception as e:
            logger.error("Hot-swap failed", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")

@app.post("/promote-shadow", tags=["Ops"], dependencies=[Depends(verify_api_key)])
async def promote_shadow_model(request: Request):
    """Instantly promotes the current shadow model to become the active model."""
    async with request.app.state.model_reload_lock:
        shadow_id = request.app.state.shadow_model_version
        if not shadow_id:
            raise HTTPException(status_code=400, detail="No shadow model loaded to promote.")
        
        old_version_id = request.app.state.active_model_version
        request.app.state.active_model_version = shadow_id
        request.app.state.shadow_model_version = None
        
        if old_version_id != shadow_id and old_version_id in request.app.state.models:
            del request.app.state.models[old_version_id]
            
        logger.info("Shadow model promoted to active.", extra={"promoted_version": shadow_id})
        return {"message": f"Successfully promoted shadow model {shadow_id} to active."}
