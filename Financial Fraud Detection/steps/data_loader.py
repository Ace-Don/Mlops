import os
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_loader(
    is_inference: bool = False, target: str = "fraud"
) -> Annotated[pd.DataFrame, "dataset"]:
    """Dataset reader and validator step.

    Reads the BankSim dataset and performs schema validation.
    
    Args:
        is_inference: If `True` target column will be removed from dataset.
        target: Name of target column in dataset.

    Returns:
        The dataset artifact as Pandas DataFrame.
    """
    # Define absolute path to the dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "Banksim", "bs140513_032310.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    dataset = pd.read_csv(dataset_path)
    
    # Schema validation
    expected_schema = {
        'step': 'int64',
        'customer': 'object',
        'age': 'object',
        'gender': 'object',
        'zipcodeOri': 'object',
        'merchant': 'object',
        'zipMerchant': 'object',
        'category': 'object',
        'amount': 'float64',
        'fraud': 'int64'
    }
    
    for col, expected_type in expected_schema.items():
        if col not in dataset.columns:
            raise ValueError(f"Missing expected column: {col}")
        
        # We can do a loose check or just rely on pandas dtypes
        actual_type = str(dataset[col].dtype)
        if expected_type == 'float64' and actual_type not in ['float64', 'float32']:
            logger.warning(f"Column {col} has type {actual_type}, expected {expected_type}")
        elif expected_type == 'int64' and actual_type not in ['int64', 'int32']:
             logger.warning(f"Column {col} has type {actual_type}, expected {expected_type}")
    
    # Null checks
    if dataset.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains unexpected null values during validation.")
    
    if is_inference and target in dataset.columns:
        dataset.drop(columns=target, inplace=True)
        
    logger.info(f"Dataset with {len(dataset)} records loaded and validated!")
    return dataset
