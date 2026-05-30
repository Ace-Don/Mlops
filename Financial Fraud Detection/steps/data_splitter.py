from typing import Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def fraud_data_splitter(
    dataset: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 10,
) -> Tuple[
    Annotated[pd.DataFrame, "fraud_dataset_trn"],
    Annotated[pd.DataFrame, "fraud_dataset_tst"],
]:
    """Fraud dataset splitter step.

    Splits the engineered dataset into stratified train and test sets.

    Split settings (from notebook Cells 51-52):
      - test_size   = 0.3   → 178 393 test rows, 416 250 train rows
      - random_state = 10
      - shuffle     = True  (default in train_test_split)

    Stratification is NOT applied here because the notebook does not stratify
    the split — only the SMOTE step within model training corrects imbalance.

    MLflow tracking experiment: 'fraud_detection_feature_engineering'

    Args:
        dataset: Engineered DataFrame (output of fraud_feature_engineer).
        test_size: Fraction of data to reserve for testing.
        random_state: Reproducibility seed — default 10 matches the notebook.

    Returns:
        Tuple of (train DataFrame, test DataFrame).
    """
    mlflow.set_experiment("fraud_detection_feature_engineering")
    with mlflow.start_run(run_name="fraud_data_split"):

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("train_size", round(1 - test_size, 4))
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("total_rows", len(dataset))

        dataset_trn, dataset_tst = train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        # Ensure column names are preserved after the split
        dataset_trn = pd.DataFrame(dataset_trn, columns=dataset.columns)
        dataset_tst = pd.DataFrame(dataset_tst, columns=dataset.columns)

        mlflow.log_metric("train_rows", len(dataset_trn))
        mlflow.log_metric("test_rows", len(dataset_tst))
        mlflow.log_metric("train_pct", round(len(dataset_trn) / len(dataset) * 100, 2))

    logger.info(
        f"Split complete — Train: {len(dataset_trn):,} rows, "
        f"Test: {len(dataset_tst):,} rows."
    )
    return dataset_trn, dataset_tst
