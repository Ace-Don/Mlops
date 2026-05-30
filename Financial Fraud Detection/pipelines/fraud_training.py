from typing import Optional
from uuid import UUID

from pipelines import fraud_feature_engineering_pipeline
from steps import (
    fraud_model_evaluator,
    fraud_model_promoter,
    fraud_model_trainer,
)
from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def fraud_training_pipeline(
    train_dataset_id: Optional[UUID] = None,
    test_dataset_id: Optional[UUID] = None,
    target: str = "fraud",
    model_type: str = "lr",
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
):
    """Fraud Model Training Pipeline.

    Trains and evaluates a fraud detection model.  Can either:
      a) Run the feature engineering pipeline first (default, when no dataset
         IDs are provided), or
      b) Re-use previously versioned train/test artifacts (pass UUIDs) —
         enabling reproducible re-training on the exact same split.

    Steps:
      1. [Optional] fraud_feature_engineering_pipeline
      2. fraud_model_trainer   — SMOTE + StandardScaler + classifier
      3. fraud_model_evaluator — accuracy, ROC-AUC, F1, precision, recall
      4. fraud_model_promoter  — promote to 'production' if better than current

    Args:
        train_dataset_id: UUID of a versioned 'fraud_preprocessed_dataset_trn'
            artifact.  If None, the feature engineering pipeline is run first.
        test_dataset_id:  UUID of a versioned 'fraud_preprocessed_dataset_tst'
            artifact.  Must be set if train_dataset_id is set.
        target:           Binary fraud target column name.
        model_type:       One of 'lr', 'xgb', 'catboost', 'lightgbm'.
        min_train_accuracy: Warn if train accuracy drops below this.
        min_test_accuracy:  Warn if test accuracy drops below this.
    """
    if train_dataset_id is None or test_dataset_id is None:
        dataset_trn, dataset_tst = fraud_feature_engineering_pipeline()
    else:
        client = Client()
        dataset_trn = client.get_artifact_version(name_id_or_prefix=train_dataset_id)
        dataset_tst = client.get_artifact_version(name_id_or_prefix=test_dataset_id)

    model = fraud_model_trainer(
        dataset_trn=dataset_trn,
        model_type=model_type,
        target=target,
    )

    acc = fraud_model_evaluator(
        model=model,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
        min_train_accuracy=min_train_accuracy,
        min_test_accuracy=min_test_accuracy,
    )

    fraud_model_promoter(accuracy=acc)
