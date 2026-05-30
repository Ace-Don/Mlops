from typing import List, Optional, Tuple

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Annotated
from zenml import log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)

# These are the three categorical columns in the Banksim dataset.
# Hardcoded here because the schema is fixed and well-documented.
CATEGORICAL_COLS = ["age", "gender", "category"]


@step
def fraud_data_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str = "fraud",
    random_state: int = 10,
) -> Tuple[
    Annotated[pd.DataFrame, "fraud_preprocessed_dataset_trn"],
    Annotated[pd.DataFrame, "fraud_preprocessed_dataset_tst"],
    Annotated[Pipeline, "fraud_preprocess_pipeline"],
]:
    """Fraud data preprocessing step.

    Applies the exact preprocessing sequence from the notebook (Cells 44-46):

      1. Strip embedded quote characters from categorical columns (age, gender,
         category) — the CSV values are stored as "'F'" etc. with single quotes.
         Applied to BOTH train and test before fitting anything.

      2. One-Hot Encode the three categorical columns using:
             OneHotEncoder(sparse_output=False, drop='first')
         The encoder is FIT on train data only, then TRANSFORM is applied to
         both splits (preventing data leakage).

      3. Build the final DataFrames by concatenating the numeric/count columns
         (amount, count_1_day, count_7_days, count_30_days) with the encoded
         categorical columns — and appending the target column UNCHANGED.

    The sklearn Pipeline wrapping the ColumnTransformer is saved as an artifact
    ('fraud_preprocess_pipeline') so the inference pipeline can apply the exact
    same transformation without re-fitting (inference consistency — req. 6).

    The target column ('fraud') is NEVER passed through any transformer.

    MLflow tracking experiment: 'fraud_detection_feature_engineering'

    Args:
        dataset_trn: Train split from fraud_data_splitter.
        dataset_tst: Test split from fraud_data_splitter.
        target: Name of the binary fraud target column (default 'fraud').
        random_state: Reproducibility seed stored in artifact metadata.

    Returns:
        Tuple of (preprocessed train DF, preprocessed test DF, fitted Pipeline).
    """
    mlflow.set_experiment("fraud_detection_feature_engineering")
    with mlflow.start_run(run_name="fraud_data_preprocessing"):

        mlflow.log_param("target", target)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("categorical_cols", CATEGORICAL_COLS)
        mlflow.log_param("ohe_drop", "first")

        # ── Step 1: Strip embedded quotes from categoricals ───────────────────
        # Notebook Cell 44: df[col].astype(str).str.replace(r"[\"']", "", regex=True)
        categorical_present = [c for c in CATEGORICAL_COLS if c in dataset_trn.columns]
        for col in categorical_present:
            dataset_trn[col] = dataset_trn[col].astype(str).str.replace(r"[\"']", "", regex=True)
            dataset_tst[col] = dataset_tst[col].astype(str).str.replace(r"[\"']", "", regex=True)
        logger.info(f"Quote-stripped categorical columns: {categorical_present}")

        # ── Step 2: Separate target from features ─────────────────────────────
        # Target is extracted before the pipeline so it is NEVER transformed.
        y_trn = dataset_trn[target].copy()
        y_tst = dataset_tst[target].copy()
        X_trn = dataset_trn.drop(columns=[target])
        X_tst = dataset_tst.drop(columns=[target])

        # ── Step 3: Build and fit the OHE ColumnTransformer Pipeline ─────────
        # Notebook Cell 46: OneHotEncoder(sparse_output=False, drop='first')
        # Passthrough retains numeric columns (amount, count_1_day, etc.) as-is.
        ohe = OneHotEncoder(sparse_output=False, drop="first")
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", ohe, categorical_present),
            ],
            remainder="passthrough",  # keeps amount, count_1/7/30_day unchanged
        )
        preprocess_pipeline = Pipeline([("preprocessor", preprocessor)])

        X_trn_transformed = preprocess_pipeline.fit_transform(X_trn)
        X_tst_transformed = preprocess_pipeline.transform(X_tst)

        # ── Step 4: Rebuild DataFrames with proper column names ───────────────
        # Derive OHE column names (drop='first' removes one category per column)
        ohe_feature_names = preprocess_pipeline.named_steps["preprocessor"].get_feature_names_out()
        # Strip the "cat__" / "remainder__" prefixes added by ColumnTransformer
        cleaned_names = [n.split("__", 1)[-1] for n in ohe_feature_names]

        dataset_trn = pd.DataFrame(X_trn_transformed, columns=cleaned_names, index=X_trn.index)
        dataset_tst = pd.DataFrame(X_tst_transformed, columns=cleaned_names, index=X_tst.index)

        # Re-attach the unmodified target column
        dataset_trn[target] = y_trn.values
        dataset_tst[target] = y_tst.values

        mlflow.log_param("train_rows", len(dataset_trn))
        mlflow.log_param("test_rows", len(dataset_tst))
        mlflow.log_param("output_features", len(cleaned_names))

    # ── Save metadata for inference pipeline lookup ───────────────────────────
    # The inference pipeline uses this to know the target column name and
    # random_state so it can reconstruct/load the correct artifact.
    log_metadata(
        metadata={"random_state": random_state, "target": target},
        artifact_name="fraud_preprocess_pipeline",
        infer_artifact=True,
    )

    logger.info(
        f"Preprocessing complete. "
        f"Train: {dataset_trn.shape}, Test: {dataset_tst.shape}. "
        f"Features: {len(cleaned_names)}"
    )
    return dataset_trn, dataset_tst, preprocess_pipeline
