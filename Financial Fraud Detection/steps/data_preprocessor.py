from typing import List, Optional, Tuple

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing_extensions import Annotated
from zenml import log_metadata, step


@step
def data_preprocessor(
    random_state: int,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    drop_na: Optional[bool] = None,
    normalize: Optional[bool] = None,
    drop_columns: Optional[List[str]] = None,
    target: Optional[str] = "fraud",
) -> Tuple[
    Annotated[pd.DataFrame, "dataset_trn"],
    Annotated[pd.DataFrame, "dataset_tst"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    """Data preprocessor step.

    Encodes categorical features and applies scaling.
    """
    mlflow.set_experiment("feature_engineering")
    mlflow.start_run(run_name="data_preprocessing")
    
    mlflow.log_param("target", target)
    
    categorical_cols = ['age', 'gender', 'category']
    # Verify these columns exist, as they might have been dropped or renamed
    categorical_cols = [c for c in categorical_cols if c in dataset_trn.columns]
    
    numeric_cols = [c for c in dataset_trn.columns if c not in categorical_cols and c != target]
    
    transformers = []
    if len(categorical_cols) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    
    if normalize and len(numeric_cols) > 0:
        transformers.append(('num', MinMaxScaler(), numeric_cols))
    else:
        transformers.append(('num', 'passthrough', numeric_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])
    
    # Separate target from features if it exists
    y_trn = dataset_trn[target] if target in dataset_trn.columns else None
    y_tst = dataset_tst[target] if target in dataset_tst.columns else None
    
    if target in dataset_trn.columns:
        X_trn = dataset_trn.drop(columns=[target])
    else:
        X_trn = dataset_trn
        
    if target in dataset_tst.columns:
        X_tst = dataset_tst.drop(columns=[target])
    else:
        X_tst = dataset_tst
        
    # Fit and transform
    X_trn_transformed = preprocess_pipeline.fit_transform(X_trn)
    X_tst_transformed = preprocess_pipeline.transform(X_tst)
    
    # Reconstruct DataFrames
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Remove the 'cat__' or 'num__' prefixes from ColumnTransformer
        feature_names = [f.split('__', 1)[-1] for f in feature_names]
    except AttributeError:
        feature_names = [str(i) for i in range(X_trn_transformed.shape[1])]
    
    dataset_trn = pd.DataFrame(X_trn_transformed, columns=feature_names, index=X_trn.index)
    dataset_tst = pd.DataFrame(X_tst_transformed, columns=feature_names, index=X_tst.index)
        
    if y_trn is not None:
        dataset_trn[target] = y_trn
    if y_tst is not None:
        dataset_tst[target] = y_tst

    # Log metadata so we can load it in the inference pipeline
    log_metadata(
        metadata={"random_state": random_state, "target": target},
        artifact_name="preprocess_pipeline",
        infer_artifact=True,
    )
    
    mlflow.log_param("train_dataset_rows", len(dataset_trn))
    mlflow.log_param("test_dataset_rows", len(dataset_tst))
    mlflow.log_param("train_dataset_cols", len(dataset_trn.columns))
    
    mlflow.end_run()
    return dataset_trn, dataset_tst, preprocess_pipeline
