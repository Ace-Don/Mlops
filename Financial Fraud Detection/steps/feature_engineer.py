import mlflow
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


# ── Rolling-window helper functions ──────────────────────────────────────────
# Exact implementations from the notebook (Cell 28).
# Each function operates on a per-customer group sorted by date and computes
# how many transactions that customer made in the preceding N days (excluding
# the current row itself, hence the -1).

def last_1_day_transaction_count(df_group: pd.DataFrame) -> pd.DataFrame:
    """Count transactions by the same customer in the past 1 day.

    Parameters:
        df_group: Customer-level sub-DataFrame sorted by date index.

    Returns:
        df_group with new column 'count_1_day'.
    """
    temp = pd.Series(df_group.index, index=df_group.date, name="count_1_day").sort_index()
    count_1_day = temp.rolling("1d").count() - 1
    count_1_day.index = temp.values
    df_group["count_1_day"] = count_1_day.reindex(df_group.index)
    return df_group


def last_7_days_transaction_count(df_group: pd.DataFrame) -> pd.DataFrame:
    """Count transactions by the same customer in the past 7 days.

    Parameters:
        df_group: Customer-level sub-DataFrame sorted by date index.

    Returns:
        df_group with new column 'count_7_days'.
    """
    temp = pd.Series(df_group.index, index=df_group.date, name="count_7_days").sort_index()
    count_7_days = temp.rolling("7d").count() - 1
    count_7_days.index = temp.values
    df_group["count_7_days"] = count_7_days.reindex(df_group.index)
    return df_group


def last_30_days_transaction_count(df_group: pd.DataFrame) -> pd.DataFrame:
    """Count transactions by the same customer in the past 30 days.

    Parameters:
        df_group: Customer-level sub-DataFrame sorted by date index.

    Returns:
        df_group with new column 'count_30_days'.
    """
    temp = pd.Series(df_group.index, index=df_group.date, name="count_30_days").sort_index()
    count_30_days = temp.rolling("30d").count() - 1
    count_30_days.index = temp.values
    df_group["count_30_days"] = count_30_days.reindex(df_group.index)
    return df_group


# ── ZenML Step ────────────────────────────────────────────────────────────────

@step
def fraud_feature_engineer(
    dataset: pd.DataFrame,
) -> Annotated[pd.DataFrame, "fraud_engineered_dataset"]:
    """Fraud feature engineering step.

    Transforms the raw Banksim transaction data into a model-ready DataFrame by:
      1. Dropping single-valued zip-code columns (zipcodeOri, zipMerchant) —
         both have exactly 1 unique value and carry no information (notebook Cell 16).
      2. Converting the 'step' column (simulation day integer) into a proper
         datetime (Jan 1 2020 + step days), renamed to 'date' (Cells 22-24).
      3. Sorting by customer + date and computing rolling 1-day, 7-day, and
         30-day transaction counts per customer (Cells 28-30).
      4. Dropping the identity columns 'customer', 'merchant', and the now-
         used 'date' column so the model never sees raw IDs or datetimes
         (Cells 32-34).

    The output retains 8 features:
      - age, gender, category  (categorical — encoded in the next step)
      - amount                  (continuous)
      - count_1_day, count_7_days, count_30_days  (engineered continuous)
      - fraud                   (target — NOT touched by any transformer)

    MLflow tracking experiment: 'fraud_detection_feature_engineering'

    Args:
        dataset: Raw transaction DataFrame from fraud_data_loader.

    Returns:
        Engineered DataFrame with 594 643 rows × 8 columns, tracked as
        'fraud_engineered_dataset'.
    """
    mlflow.set_experiment("fraud_detection_feature_engineering")
    with mlflow.start_run(run_name="fraud_feature_engineering"):

        mlflow.log_param("input_rows", len(dataset))
        mlflow.log_param("input_cols", len(dataset.columns))

        # ── Step 1: Drop constant zip-code columns ────────────────────────────
        cols_to_drop = [c for c in ["zipMerchant", "zipcodeOri"] if c in dataset.columns]
        if cols_to_drop:
            dataset = dataset.drop(cols_to_drop, axis=1)
            logger.info(f"Dropped constant columns: {cols_to_drop}")

        # ── Step 2: Convert 'step' → 'date' datetime ─────────────────────────
        # 1577836800 = Unix timestamp for Jan 1, 2020 00:00:00 UTC.
        # Each step represents one simulation day → multiply by 86 400 seconds.
        dataset["step"] = 1577836800 + dataset["step"] * 3600 * 24
        dataset["step"] = pd.to_datetime(dataset["step"], unit="s")
        dataset = dataset.rename(columns={"step": "date"})
        logger.info("Converted 'step' column to datetime 'date'.")

        # ── Step 3: Rolling transaction counts ────────────────────────────────
        dataset = dataset.sort_values(["customer", "date"])

        # groupby().apply() works in both older and newer pandas.
        dataset = dataset.groupby("customer").apply(last_1_day_transaction_count).reset_index(drop=True)
        dataset = dataset.groupby("customer").apply(last_7_days_transaction_count).reset_index(drop=True)
        dataset = dataset.groupby("customer").apply(last_30_days_transaction_count).reset_index(drop=True)
        logger.info("Computed rolling 1-day, 7-day, and 30-day transaction counts.")

        # ── Step 4: Drop identity / datetime columns ──────────────────────────
        dataset = dataset.drop(["customer", "merchant", "date"], axis=1)
        logger.info("Dropped 'customer', 'merchant', and 'date' columns.")

        mlflow.log_param("output_rows", len(dataset))
        mlflow.log_param("output_cols", len(dataset.columns))
        mlflow.log_param("engineered_columns", list(dataset.columns))

    logger.info(
        f"Feature engineering complete. "
        f"Output shape: {dataset.shape}. "
        f"Columns: {list(dataset.columns)}"
    )
    return dataset
