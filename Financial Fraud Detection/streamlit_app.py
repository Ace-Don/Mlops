import streamlit as st
import requests
import redis
import pandas as pd
import os
from dotenv import load_dotenv

# Load config
load_dotenv()
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

st.set_page_config(page_title="Fraud MLOps", page_icon="🛡️", layout="wide")

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("X-API-Key", os.getenv("API_KEY", "super-secret-key"), type="password")
    st.markdown("---")
    st.write("### 🔗 Quick Links")
    st.markdown("[Swagger API Docs](http://localhost:8000/docs)")
    st.markdown("[Prometheus Metrics](http://localhost:9090)")
    st.markdown("[Grafana Dashboard](http://localhost:3000)")
    st.markdown("[MLflow Tracking](http://localhost:5000)")

def api_post(endpoint, payload=None, params=None, auth=False):
    headers = {"X-API-Key": api_key} if auth else {}
    try:
        return requests.post(f"{FASTAPI_URL}{endpoint}", json=payload, params=params, headers=headers)
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None

# ==========================================
# HEADER
# ==========================================
st.title("🛡️ Financial Fraud Control Panel")
st.markdown("A unified dashboard for serving, shadow deployments, and feature store monitoring.")
st.markdown("---")

# ==========================================
# MAIN TABS
# ==========================================
t_predict, t_ops, t_redis = st.tabs(["🔮 Live Predictions & Feedback", "🚀 Model Operations", "💾 Redis Features"])

# ------------------------------------------
# TAB 1: PREDICTIONS & FEEDBACK
# ------------------------------------------
with t_predict:
    col_p1, col_p2 = st.columns([1, 1], gap="large")
    
    with col_p1:
        st.subheader("Run Prediction")
        with st.form("predict_form"):
            customer_id = st.text_input("Customer ID", "cust_12345")
            c1, c2 = st.columns(2)
            age = c1.selectbox("Age Group", ["0", "1", "2", "3", "4", "5", "6", "U"], index=3)
            gender = c2.selectbox("Gender", ["M", "F", "E", "U"], index=0)
            category = st.selectbox("Category", ["es_transportation", "es_food", "es_health", "es_shopping", "es_tech"], index=0)
            amount = st.number_input("Amount ($)", min_value=0.0, value=25.50)
            
            submitted = st.form_submit_button("Analyze Transaction", use_container_width=True)
            
        if submitted:
            resp = api_post("/predict", payload={"customer_id": customer_id, "age": age, "gender": gender, "category": category, "amount": amount})
            if resp and resp.status_code == 200:
                res = resp.json()
                if res.get("diagnosis_code") == 1:
                    st.error(f"🚨 FRAUD DETECTED ({res.get('confidence')*100:.1f}% Confidence)")
                else:
                    st.success(f"✅ LEGITIMATE ({res.get('confidence')*100:.1f}% Confidence)")
            else:
                st.error("Prediction failed. Check API status.")

    with col_p2:
        st.subheader("Log Ground Truth Feedback")
        st.info("Update logs with verified outcomes to improve future training.")
        with st.form("feedback_form"):
            pred_id = st.number_input("Prediction ID (from MySQL)", min_value=1, value=1)
            actual = st.radio("Verified Outcome", ["Legitimate", "Fraudulent"], horizontal=True)
            fb_submitted = st.form_submit_button("Submit Label", use_container_width=True)
            
        if fb_submitted:
            val = 1 if actual == "Fraudulent" else 0
            resp = api_post("/feedback", payload={"prediction_id": pred_id, "actual_fraud": val})
            if resp and resp.status_code == 200:
                st.success(f"Label updated for Prediction #{pred_id}!")

# ------------------------------------------
# TAB 2: MODEL OPERATIONS
# ------------------------------------------
with t_ops:
    st.subheader("Deployment & Retraining")
    
    col_o1, col_o2, col_o3 = st.columns(3)
    
    with col_o1:
        st.write("**1. Fetch Staging Model**")
        st.caption("Load a candidate model silently alongside Production.")
        if st.button("Load Shadow Model", use_container_width=True):
            with st.spinner("Fetching from ZenML..."):
                resp = api_post("/reload-model", params={"load_as_shadow": "true"}, auth=True)
                if resp and resp.status_code == 200:
                    st.success("Shadow Model Loaded!")
                elif resp and "No model version found" in resp.text:
                    st.warning("⚠️ No 'Staging' model exists yet. You must trigger a retrain loop that beats the current production accuracy first!")
                else:
                    st.error(f"Failed: {resp.text if resp else 'Connection Error'}")

    with col_o2:
        st.write("**2. Promote to Production**")
        st.caption("Swap the shadow model to Active serving instantly.")
        if st.button("Promote Shadow", use_container_width=True):
            resp = api_post("/promote-shadow", auth=True)
            if resp and resp.status_code == 200:
                st.success("Promoted successfully!")
            else:
                st.error(f"Failed: {resp.text if resp else 'Connection Error'}")

    with col_o3:
        st.write("**3. Retrain Pipeline**")
        st.caption("Offload training job to background Celery worker.")
        
        model_options = {
            "Logistic Regression": "lr",
            "XGBoost": "xgb",
            "CatBoost": "catboost",
            "LightGBM": "lightgbm"
        }
        selected_model_name = st.selectbox("Select Algorithm", list(model_options.keys()))
        selected_model_id = model_options[selected_model_name]
        
        if st.button("Trigger Retraining", use_container_width=True):
            resp = api_post("/retrain", params={"model_type": selected_model_id}, auth=True)
            if resp and resp.status_code == 200:
                st.success(f"Queued {selected_model_name} in Celery!")
            else:
                st.error("Failed to queue job.")

# ------------------------------------------
# TAB 3: REDIS FEATURE STORE
# ------------------------------------------
with t_redis:
    st.subheader("Live Redis Cache")
    st.caption("Sub-millisecond rolling transaction counts for all active customers.")
    
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()
        
        search = st.text_input("Filter Customers (e.g., customer:*)", "customer:*")
        keys = r.keys(search)
        
        if keys:
            data = []
            for k in keys[:50]:  # Limit to 50 for fast UI rendering
                f = r.hgetall(k)
                f["Customer"] = k
                data.append(f)
                
            df = pd.DataFrame(data).fillna(0)
            st.dataframe(df, use_container_width=True, hide_index=True)
            if len(keys) > 50:
                st.caption(f"Showing 50 of {len(keys)} records.")
        else:
            st.info("Cache is empty. Run a prediction to populate.")
    except Exception as e:
        st.error("Could not connect to Redis Feature Store container on port 6379.")
