from typing import Any, Optional

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing_extensions import Annotated
from zenml import log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def fraud_model_evaluator(
    model: Any,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str = "fraud",
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
) -> float:
    """Fraud model evaluation step.

    Evaluates the trained model against the hold-out test set using the five
    metrics reported in the notebook (Cell 64):

      - Accuracy           (binary, straightforward)
      - ROC-AUC            (probability-based; uses predict_proba if available)
      - F1-score   (macro) — matches notebook: f1_score(y_test, pred, average='macro')
      - Precision  (macro) — matches notebook
      - Recall     (macro) — matches notebook

    NOTE: The notebook uses average='macro' for F1/Precision/Recall, unlike the
    template which uses average='weighted'.  We follow the notebook exactly.

    Metrics and evaluation parameters are:
      - Logged to MLflow under experiment 'fraud_detection_model_training'
      - Stored as ZenML artifact run_metadata on the 'fraud_trained_model'
        artifact so the model_promoter can compare with the production version.

    MLflow tracking experiment: 'fraud_detection_model_training'

    Args:
        model: The fitted imblearn Pipeline from fraud_model_trainer.
        dataset_trn: Preprocessed training DataFrame (with target column).
        dataset_tst: Preprocessed test DataFrame (with target column).
        target: Name of the binary fraud target column.
        min_train_accuracy: Log a warning if train accuracy falls below this.
        min_test_accuracy: Log a warning if test accuracy falls below this.

    Returns:
        Test accuracy (float) — passed to fraud_model_promoter.
    """
    mlflow.set_experiment("fraud_detection_model_training")
    with mlflow.start_run(run_name="fraud_model_evaluation"):

        mlflow.log_param("target", target)
        mlflow.log_param("min_train_accuracy", min_train_accuracy)
        mlflow.log_param("min_test_accuracy", min_test_accuracy)

        X_trn = dataset_trn.drop(columns=[target])
        y_trn = dataset_trn[target]
        X_tst = dataset_tst.drop(columns=[target])
        y_tst = dataset_tst[target]

        y_trn_pred = model.predict(X_trn)
        y_tst_pred = model.predict(X_tst)

        # ── Core metrics (notebook Cell 64) ──────────────────────────────────
        trn_acc = accuracy_score(y_trn, y_trn_pred)
        tst_acc = accuracy_score(y_tst, y_tst_pred)
        tst_f1 = f1_score(y_tst, y_tst_pred, average="macro", zero_division=0)
        tst_precision = precision_score(y_tst, y_tst_pred, average="macro", zero_division=0)
        tst_recall = recall_score(y_tst, y_tst_pred, average="macro", zero_division=0)

        # ROC-AUC — use predict_proba if available, else skip gracefully
        tst_roc_auc = None
        try:
            y_tst_proba = model.predict_proba(X_tst)[:, 1]
            tst_roc_auc = roc_auc_score(y_tst, y_tst_proba)
        except (AttributeError, ValueError):
            logger.warning("ROC-AUC could not be computed (model lacks predict_proba).")

        logger.info(f"Train accuracy  = {trn_acc * 100:.2f}%")
        logger.info(f"Test accuracy   = {tst_acc * 100:.2f}%")
        logger.info(f"Test F1 (macro) = {tst_f1 * 100:.2f}%")
        logger.info(f"Test Precision  = {tst_precision * 100:.2f}%")
        logger.info(f"Test Recall     = {tst_recall * 100:.2f}%")
        if tst_roc_auc is not None:
            logger.info(f"Test ROC-AUC    = {tst_roc_auc:.6f}")

        # ── Log to MLflow ─────────────────────────────────────────────────────
        mlflow.log_metric("fraud_train_accuracy", trn_acc)
        mlflow.log_metric("fraud_test_accuracy", tst_acc)
        mlflow.log_metric("fraud_test_f1_macro", tst_f1)
        mlflow.log_metric("fraud_test_precision_macro", tst_precision)
        mlflow.log_metric("fraud_test_recall_macro", tst_recall)
        if tst_roc_auc is not None:
            mlflow.log_metric("fraud_test_roc_auc", tst_roc_auc)

        # ── Threshold warnings ────────────────────────────────────────────────
        messages = []
        if trn_acc < min_train_accuracy:
            messages.append(
                f"Train accuracy {trn_acc*100:.2f}% is below threshold {min_train_accuracy*100:.2f}%"
            )
        if tst_acc < min_test_accuracy:
            messages.append(
                f"Test accuracy {tst_acc*100:.2f}% is below threshold {min_test_accuracy*100:.2f}%"
            )
        for msg in messages:
            logger.warning(msg)
        if messages:
            mlflow.log_param("fraud_eval_warnings", " | ".join(messages))

    # ── Store metadata on the fraud_trained_model artifact ───────────────────
    # The model_promoter retrieves 'test_accuracy' from run_metadata to
    # compare this version against the current production model.
    client = Client()
    latest_model_artifact = client.get_artifact_version("fraud_trained_model")
    metadata_payload = {
        "train_accuracy": float(trn_acc),
        "test_accuracy": float(tst_acc),
        "test_f1_macro": float(tst_f1),
        "test_precision_macro": float(tst_precision),
        "test_recall_macro": float(tst_recall),
    }
    if tst_roc_auc is not None:
        metadata_payload["test_roc_auc"] = float(tst_roc_auc)

    log_metadata(
        metadata=metadata_payload,
        artifact_version_id=latest_model_artifact.id,
    )

    return float(tst_acc)
