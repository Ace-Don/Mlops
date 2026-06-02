"""
NOTE: This file has been commented out and deprecated.

When it could be useful:
This ZenML pipeline and its associated steps are useful for offline, macro-batch 
scoring. For example, if you need to run inference on millions of rows directly 
from a data warehouse overnight, bypassing HTTP network overhead.

Reason for commenting out:
For the current project scope, our FastAPI service (`inference_api.py`) handles 
both real-time streaming and micro-batching natively. It also implements a full 
continuous learning loop (logging predictions, accepting ground truth feedback, 
detecting drift, and automatically triggering retraining). Because the API covers 
all of our inference needs and relies directly on the model artifacts, this 
legacy simulation pipeline is redundant.
"""

# from typing import Any
# 
# import mlflow
# import pandas as pd
# from typing_extensions import Annotated
# from zenml import step
# from zenml.logger import get_logger
# 
# logger = get_logger(__name__)
# 
# 
# @step
# def fraud_inference_predict(
#     model: Any,
#     dataset_inf: pd.DataFrame,
# ) -> Annotated[pd.Series, "fraud_predictions"]:
#     """Fraud inference prediction step.
# 
#     Runs batch prediction on the preprocessed inference DataFrame and returns
#     a Series of predicted labels (0 = legitimate, 1 = fraud).
# 
#     Also logs:
#       - Prediction confidence (avg max probability if model has predict_proba)
#       - Count of predicted fraud vs. legitimate transactions
#       - Fraud rate (%) as an MLflow metric for monitoring
# 
#     MLflow tracking experiment: 'fraud_detection_inference'
# 
#     Args:
#         model: The production imblearn Pipeline (from fraud_model_promoter stage).
#         dataset_inf: Preprocessed inference DataFrame (no target column).
# 
#     Returns:
#         pd.Series of integer predictions (0 or 1) named 'fraud_predicted'.
#     """
#     mlflow.set_experiment("fraud_detection_inference")
#     with mlflow.start_run(run_name="fraud_batch_predictions"):
# 
#         mlflow.log_param("inference_samples", dataset_inf.shape[0])
#         mlflow.log_param("inference_features", dataset_inf.shape[1])
# 
#         predictions = model.predict(dataset_inf)
# 
#         # ── Confidence / probability logging ──────────────────────────────────
#         try:
#             proba = model.predict_proba(dataset_inf)
#             avg_confidence = proba.max(axis=1).mean()
#             mlflow.log_metric("fraud_avg_prediction_confidence", avg_confidence)
#             logger.info(f"Average prediction confidence: {avg_confidence:.4f}")
#         except AttributeError:
#             logger.warning("Model does not support predict_proba — skipping confidence log.")
# 
#         predictions = pd.Series(predictions, name="fraud_predicted")
# 
#         # ── Fraud statistics ──────────────────────────────────────────────────
#         fraud_count = int((predictions == 1).sum())
#         legit_count = int((predictions == 0).sum())
#         fraud_rate = fraud_count / len(predictions) * 100
# 
#         mlflow.log_metric("fraud_predictions_total", len(predictions))
#         mlflow.log_metric("fraud_predicted_legitimate", legit_count)
#         mlflow.log_metric("fraud_predicted_fraudulent", fraud_count)
#         mlflow.log_metric("fraud_predicted_rate_pct", round(fraud_rate, 4))
# 
#     logger.info(
#         f"Predictions complete — {len(predictions):,} transactions | "
#         f"Fraudulent: {fraud_count:,} ({fraud_rate:.2f}%) | "
#         f"Legitimate: {legit_count:,}"
#     )
#     return predictions
