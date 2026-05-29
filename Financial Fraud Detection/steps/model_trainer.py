import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Optional, Any
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def model_trainer(
    dataset_trn: pd.DataFrame,
    model_type: str = "lr",
    target: Optional[str] = "fraud",
) -> Annotated[Any, ArtifactConfig(name="trained_model", is_model_artifact=True)]:
    """Configure and train a model on the training dataset.

    Supports Logistic Regression, XGBoost, CatBoost, and LightGBM.
    """
    mlflow.set_experiment("model_training")
    mlflow.start_run(run_name=f"train_model_{model_type}")
    
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("target", target)
    
    X_train = dataset_trn.drop(columns=[target])
    y_train = dataset_trn[target]
    
    if model_type == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    elif model_type == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(verbose=0)
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier()
    else:
        mlflow.end_run()
        raise ValueError(f"Unknown model type {model_type}")

    logger.info(f"Training model {model_type}...")
    model.fit(X_train, y_train)
    
    if model_type == "lr":
        mlflow.sklearn.log_model(model, "model")
    elif model_type == "xgb":
        import mlflow.xgboost
        mlflow.xgboost.log_model(model, "model")
    elif model_type == "catboost":
        try:
            import mlflow.catboost
            mlflow.catboost.log_model(model, "model")
        except AttributeError:
            mlflow.sklearn.log_model(model, "model")
    elif model_type == "lightgbm":
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(model, "model")
    
    mlflow.log_param("training_samples", len(dataset_trn))
    mlflow.log_param("training_features", len(dataset_trn.columns) - 1)
    
    mlflow.end_run()
    return model
