from steps import (
    fraud_data_loader,
    fraud_data_preprocessor,
    fraud_data_splitter,
    fraud_feature_engineer,
)
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def fraud_feature_engineering_pipeline(
    target: str = "fraud",
    test_size: float = 0.3,
    random_state: int = 10,
):
    """Fraud Feature Engineering Pipeline.

    Prepares the Banksim transaction dataset for model training:
      1. fraud_data_loader        — loads & schema-validates the CSV
      2. fraud_feature_engineer   — rolling window features + column drops
      3. fraud_data_splitter      — 70/30 split (random_state=10)
      4. fraud_data_preprocessor  — quote strip + OHE(drop='first')

    The processed train/test DataFrames and the fitted OHE Pipeline are tracked
    as ZenML artifacts:
      - 'fraud_preprocessed_dataset_trn'
      - 'fraud_preprocessed_dataset_tst'
      - 'fraud_preprocess_pipeline'

    Args:
        target:       Name of the binary fraud target column.
        test_size:    Fraction of data reserved for testing (notebook: 0.3).
        random_state: Reproducibility seed (notebook: 10).
    """
    raw_data = fraud_data_loader(target=target)
    engineered_data = fraud_feature_engineer(dataset=raw_data)
    dataset_trn, dataset_tst = fraud_data_splitter(
        dataset=engineered_data,
        test_size=test_size,
        random_state=random_state,
    )
    dataset_trn, dataset_tst, _ = fraud_data_preprocessor(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
        random_state=random_state,
    )
    return dataset_trn, dataset_tst
