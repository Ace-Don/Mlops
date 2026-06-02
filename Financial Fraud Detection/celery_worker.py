"""
Celery Worker for Financial Fraud Detection
Executes heavy, decoupled tasks asynchronously (like triggering ZenML pipelines).
"""

import os
import subprocess
import logging
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("celery_worker")

app = Celery("fraud_tasks", broker=BROKER_URL, backend=BROKER_URL)

@app.task(name="run_training_pipeline_task")
def run_training_pipeline_task():
    """
    Executes the ZenML training pipeline as a decoupled Celery task.
    """
    logger.info("Executing training pipeline via Celery task...")
    try:
        # We run it as a subprocess to keep the Celery worker environment clean and separate from ZenML execution
        result = subprocess.run(
            ["python", "run.py"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        logger.info(f"Training pipeline completed successfully. Output:\n{result.stdout}")
        return {"status": "success", "message": "Pipeline completed"}
    except subprocess.CalledProcessError as e:
        logger.error(f"Training pipeline failed. Error:\n{e.stderr}")
        return {"status": "error", "message": e.stderr}
    except Exception as e:
        logger.error(f"Unexpected error executing pipeline: {e}")
        return {"status": "error", "message": str(e)}
