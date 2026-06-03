"""
Scheduled Drift Monitor
Fetches recent production predictions and compares them to baseline data using Evidently.
"""

import os
import pymysql
import pandas as pd
import logging
import json
from dotenv import load_dotenv
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("drift_monitor")

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mysqlrootpassword")
MYSQL_DB = os.getenv("MYSQL_DB", "fraud_inference_db")

def fetch_production_data(limit=1000):
    """Fetches recent input features from the inference_logs table."""
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cur:
            cur.execute("SELECT input_features FROM inference_logs ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            if not rows:
                return pd.DataFrame()
            
            # Parse JSON features
            parsed_data = [json.loads(row['input_features']) for row in rows]
            return pd.DataFrame(parsed_data)
    except Exception as e:
        logger.error(f"Failed to fetch production data: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

def fetch_reference_data():
    """
    Ideally, this fetches the training dataset from ZenML or a blob store.
    For demonstration, we try to load it locally if available.
    """
    try:
        # Assuming training data might be stored locally by the pipeline or is mockable
        if os.path.exists("Banksim/bank_sim_data.csv"):
            df = pd.read_csv("Banksim/bank_sim_data.csv").head(2000)
            return df
        else:
            logger.warning("Reference dataset not found locally. Cannot perform drift check.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch reference data: {e}")
        return pd.DataFrame()

def run_drift_check():
    logger.info("Starting drift monitoring job...")
    
    current_data = fetch_production_data()
    if current_data.empty:
        logger.info("No recent production data found to check for drift.")
        return
        
    reference_data = fetch_reference_data()
    if reference_data.empty:
        return
        
    # Align columns (Evidently requires same columns)
    common_cols = list(set(current_data.columns).intersection(set(reference_data.columns)))
    if not common_cols:
         logger.warning("No overlapping columns between reference and current data.")
         return
         
    current_data = current_data[common_cols]
    reference_data = reference_data[common_cols]

    logger.info(f"Comparing {len(current_data)} recent predictions against {len(reference_data)} reference rows.")
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save the HTML report for visual dashboard access
    report.save_html("drift_report.html")
    logger.info("Evidently HTML drift report saved successfully as 'drift_report.html'.")
    
    drift_result = report.as_dict()
    dataset_drift = drift_result['metrics'][0]['result']['dataset_drift']
    
    if dataset_drift:
        logger.warning("🚨 DATA DRIFT DETECTED! 🚨")
        logger.warning("Significant data drift has been identified in recent transactions.")
        logger.warning("Action Required: Consider retraining the model.")
        # User requested to only print a warning for now, so we do not trigger the /retrain endpoint here.
    else:
        logger.info("✅ No data drift detected. Model is operating normally.")

if __name__ == "__main__":
    run_drift_check()
