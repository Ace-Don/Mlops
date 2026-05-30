from typing import Any, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def fraud_model_trainer(
    dataset_trn: pd.DataFrame,
    model_type: str = "lr",
    target: str = "fraud",
) -> Annotated[
    Any,
    ArtifactConfig(name="fraud_trained_model", artifact_type=ArtifactType.MODEL),
]:
    """Fraud model training step.

    Trains a classifier using the exact pipeline structure from the notebook
    (Cells 57, 59, 61, 63):

        SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.8)
        → StandardScaler()
        → <Classifier>

    SMOTE oversamples the minority fraud class (sampling_strategy=0.8 means the
    minority class will be brought to 80% of the majority count).  StandardScaler
    normalises features after oversampling so the scaler statistics are not
    skewed by the synthetic points.

    The whole imblearn Pipeline is returned as the model artifact so the same
    object handles inference — it simply skips the SMOTE step at predict time.

    Supported model types (notebook models):
      - 'lr'        — Logistic Regression (solver='liblinear', max_iter=1000)
      - 'xgb'       — XGBoost
      - 'catboost'  — CatBoost
      - 'lightgbm'  — LightGBM

    MLflow tracking experiment: 'fraud_detection_model_training'

    Args:
        dataset_trn: Preprocessed training DataFrame (features + target column).
        model_type:  One of 'lr', 'xgb', 'catboost', 'lightgbm'.
        target:      Name of the binary fraud target column.

    Returns:
        Fitted imblearn Pipeline saved as 'fraud_trained_model' model artifact.

    Raises:
        ValueError: If model_type is not one of the supported values.
    """
    mlflow.set_experiment("fraud_detection_model_training")
    with mlflow.start_run(run_name=f"fraud_train_{model_type}"):

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("target", target)
        mlflow.log_param("smote_random_state", 42)
        mlflow.log_param("smote_k_neighbors", 5)
        mlflow.log_param("smote_sampling_strategy", 0.8)
        mlflow.log_param("training_samples", len(dataset_trn))
        mlflow.log_param("training_features", len(dataset_trn.columns) - 1)

        X_train = dataset_trn.drop(columns=[target])
        y_train = dataset_trn[target]

        # ── Choose classifier ─────────────────────────────────────────────────
        if model_type == "lr":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                solver="liblinear",
                random_state=42,
                max_iter=1000,
            )
            mlflow.log_param("solver", "liblinear")
            mlflow.log_param("max_iter", 1000)

        elif model_type == "xgb":
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                eval_metric="logloss",
                random_state=42,
            )

        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            clf = CatBoostClassifier(verbose=0, random_state=42)

        elif model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Choose from: 'lr', 'xgb', 'catboost', 'lightgbm'."
            )

        # ── Build and fit imblearn Pipeline ───────────────────────────────────
        model = Pipeline([
            ("smote", SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.8)),
            ("scaler", StandardScaler()),
            ("model", clf),
        ])

        logger.info(f"Training fraud_{model_type} with SMOTE → StandardScaler → {clf.__class__.__name__}...")
        model.fit(X_train, y_train)

        # ── Log to MLflow ─────────────────────────────────────────────────────
        # The whole imblearn Pipeline is serialised as a sklearn-compatible model.
        mlflow.sklearn.log_model(model, f"fraud_model_{model_type}")

    logger.info(f"Model training complete for '{model_type}'.")
    return model
