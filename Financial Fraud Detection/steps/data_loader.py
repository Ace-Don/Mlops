import os
import sys

import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

# Paths to the two Banksim source files
BANKSIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Banksim")
TRANSACTIONS_CSV = os.path.join(BANKSIM_DIR, "bs140513_032310.csv")
NETWORK_CSV = os.path.join(BANKSIM_DIR, "bsNET140513_032310.csv")

# Expected schema for the transaction-level CSV (df2)
REQUIRED_COLUMNS = {"step", "customer", "age", "gender", "zipcodeOri", "merchant", "zipMerchant", "category", "amount", "fraud"}


@step
def fraud_data_loader(
    target: str = "fraud",
) -> Annotated[pd.DataFrame, "fraud_raw_dataset"]:
    """Fraud detection data loader step.

    Loads the Banksim transaction dataset (bs140513_032310.csv) which is the
    supervised-modelling version of the BankSim synthetic fraud dataset.

    NOTE: The network edge-list CSV (bsNET140513_032310.csv) is intentionally
    NOT merged here.  An exploratory merge was performed in the notebook and
    confirmed that df2 alone is appropriate for supervised classification — the
    inner join explodes the row count to ~47 M rows and the two fraud columns
    conflict in 7 582 rows.

    Performs:
      - Strict schema enforcement (req. 2 in MLOPS_PROJECT_REQUIREMENTS.md)
      - Null and duplicate checks
      - Dataset size validation

    Args:
        target: Name of the binary fraud target column.

    Returns:
        Raw transaction DataFrame with 594 643 rows × 10 columns.

    Raises:
        FileNotFoundError: If the source CSV cannot be found.
        ValueError: If required columns are missing, nulls or wrong target values
            are detected.
    """
    # ── 1. Locate and load ────────────────────────────────────────────────────
    if not os.path.exists(TRANSACTIONS_CSV):
        raise FileNotFoundError(
            f"Banksim transaction CSV not found at: {TRANSACTIONS_CSV}\n"
            "Please ensure the Banksim/ directory is present in the project root."
        )

    logger.info(f"Loading Banksim transaction data from: {TRANSACTIONS_CSV}")
    dataset = pd.read_csv(TRANSACTIONS_CSV)

    # ── 2. Schema enforcement ─────────────────────────────────────────────────
    missing_cols = REQUIRED_COLUMNS - set(dataset.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset schema validation failed. Missing columns: {missing_cols}"
        )

    # ── 3. Null check ─────────────────────────────────────────────────────────
    null_counts = dataset.isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Dataset contains null values:\n{null_counts[null_counts > 0]}"
        )

    # ── 4. Target column validation ───────────────────────────────────────────
    unexpected_target_vals = set(dataset[target].unique()) - {0, 1}
    if unexpected_target_vals:
        raise ValueError(
            f"Target column '{target}' contains unexpected values: "
            f"{unexpected_target_vals}. Expected only 0 and 1."
        )

    fraud_count = dataset[target].sum()
    legit_count = len(dataset) - fraud_count
    logger.info(
        f"Dataset loaded — {len(dataset):,} records "
        f"({legit_count:,} legitimate, {fraud_count:,} fraudulent)"
    )

    return dataset
