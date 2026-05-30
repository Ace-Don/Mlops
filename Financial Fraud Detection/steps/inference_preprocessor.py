import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def fraud_inference_preprocessor(
    dataset_inf: pd.DataFrame,
    preprocess_pipeline: Pipeline,
    target: str = "fraud",
) -> Annotated[pd.DataFrame, "fraud_inference_dataset"]:
    """Fraud inference preprocessing step.

    Applies the saved OHE sklearn Pipeline (from training) to new inference data.

    This step is the critical component of inference consistency (req. 6 in
    MLOPS_PROJECT_REQUIREMENTS.md).  The same Pipeline that was fit on training
    data is applied here — ensuring that quote-stripping and OHE produce exactly
    the same feature columns as during training.

    The inference data arrives WITHOUT the target column.  To avoid any
    ColumnTransformer mismatch, a dummy target column of 1s is temporarily
    injected, the pipeline is applied, and the dummy column is then dropped.

    MLflow tracking experiment: 'fraud_detection_inference'

    Args:
        dataset_inf: Raw inference DataFrame (no target column).
        preprocess_pipeline: The fitted sklearn Pipeline from fraud_data_preprocessor.
        target: Name of the target column (needed only to manage the dummy injection).

    Returns:
        Transformed inference DataFrame ready for model.predict().
    """
    mlflow.set_experiment("fraud_detection_inference")
    with mlflow.start_run(run_name="fraud_inference_preprocessing"):

        mlflow.log_param("inference_rows", dataset_inf.shape[0])
        mlflow.log_param("inference_input_cols", dataset_inf.shape[1])
        mlflow.log_param("target_column", target)

        # ── Strip embedded quotes from categoricals ───────────────────────────
        # Exactly mirrors the training-time quote stripping so OHE categories match.
        categorical_cols = ["age", "gender", "category"]
        for col in categorical_cols:
            if col in dataset_inf.columns:
                dataset_inf[col] = (
                    dataset_inf[col].astype(str).str.replace(r"[\"']", "", regex=True)
                )

        # ── Apply the saved pipeline ──────────────────────────────────────────
        # Inject a dummy target column so the pipeline doesn't fail on shape.
        dataset_inf[target] = 1
        dataset_inf = preprocess_pipeline.transform(dataset_inf)
        dataset_inf = dataset_inf.drop(columns=[target], errors="ignore")

        mlflow.log_metric("inference_output_cols", dataset_inf.shape[1])

    logger.info(f"Inference preprocessing complete. Shape: {dataset_inf.shape}")
    return dataset_inf
