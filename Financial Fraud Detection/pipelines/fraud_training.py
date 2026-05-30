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
    smote_k_neighbors: int = 5,
    smote_sampling_strategy: float = 0.8,
    lr_solver: str = "liblinear",
    lr_max_iter: int = 1000,
    xgb_n_estimators: int = 100,
    xgb_max_depth: int = 3,
    xgb_learning_rate: float = 0.1,
    xgb_subsample: float = 1.0,
    xgb_colsample_bytree: float = 1.0,
    xgb_scale_pos_weight: float = 1.0,
    xgb_eval_metric: str = "logloss",
    cat_iterations: int = 100,
    cat_depth: int = 3,
    cat_learning_rate: float = 0.1,
    cat_l2_leaf_reg: float = 3.0,
    lgb_n_estimators: int = 100,
    lgb_num_leaves: int = 31,
    lgb_learning_rate: float = 0.1,
    lgb_subsample: float = 1.0,
    lgb_colsample_bytree: float = 1.0,
    lgb_reg_alpha: float = 0.0,
    lgb_reg_lambda: float = 0.0,
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
        smote_k_neighbors=smote_k_neighbors,
        smote_sampling_strategy=smote_sampling_strategy,
        lr_solver=lr_solver,
        lr_max_iter=lr_max_iter,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
        xgb_scale_pos_weight=xgb_scale_pos_weight,
        xgb_eval_metric=xgb_eval_metric,
        cat_iterations=cat_iterations,
        cat_depth=cat_depth,
        cat_learning_rate=cat_learning_rate,
        cat_l2_leaf_reg=cat_l2_leaf_reg,
        lgb_n_estimators=lgb_n_estimators,
        lgb_num_leaves=lgb_num_leaves,
        lgb_learning_rate=lgb_learning_rate,
        lgb_subsample=lgb_subsample,
        lgb_colsample_bytree=lgb_colsample_bytree,
        lgb_reg_alpha=lgb_reg_alpha,
        lgb_reg_lambda=lgb_reg_lambda,
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
