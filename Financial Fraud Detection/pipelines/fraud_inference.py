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

# from steps import (
#     fraud_data_loader,
#     fraud_inference_predict,
#     fraud_inference_preprocessor,
# )
# from zenml import get_pipeline_context, pipeline
# from zenml.logger import get_logger
# 
# logger = get_logger(__name__)
# 
# 
# @pipeline
# def fraud_inference_pipeline(
#     target: str = "fraud",
# ):
#     """Fraud Inference Pipeline.
# 
#     Loads a batch of raw transaction data (inference subset sampled from the
#     Banksim CSV), preprocesses it using the saved OHE pipeline from training,
#     and runs fraud predictions using the production model.
# 
#     Artifacts are retrieved from the ZenML model version in context:
#       - 'fraud_trained_model'      — the production fraud classifier
#       - 'fraud_preprocess_pipeline' — the fitted OHE + passthrough pipeline
# 
#     Outputs:
#       - 'fraud_predictions' (pd.Series) — 0 = legitimate, 1 = fraud
# 
#     Args:
#         target: Name of the binary fraud target column (used only to manage
#                 the temporary dummy column during preprocessing).
#     """
#     # ── Retrieve production model and preprocessing pipeline ──────────────────
#     model = get_pipeline_context().model.get_artifact("fraud_trained_model")
#     preprocess_pipeline = get_pipeline_context().model.get_artifact(
#         "fraud_preprocess_pipeline"
#     )
# 
#     # ── Load inference data ───────────────────────────────────────────────────
#     # The data_loader is called with is_inference=True (to be implemented as
#     # a lightweight wrapper that returns a sample of the dataset without the
#     # target column).  For now the full loader is called and the target column
#     # is removed inside the inference_preprocessor to keep the pipeline simple.
#     df_inference = fraud_data_loader(target=target)
# 
#     # ── Preprocess using saved pipeline ──────────────────────────────────────
#     df_inference = fraud_inference_preprocessor(
#         dataset_inf=df_inference,
#         preprocess_pipeline=preprocess_pipeline,
#         target=target,
#     )
# 
#     # ── Predict ───────────────────────────────────────────────────────────────
#     fraud_inference_predict(
#         model=model,
#         dataset_inf=df_inference,
#     )
