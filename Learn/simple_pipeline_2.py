from typing import Annotated
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

from zenml import pipeline, step, Model
from zenml.config import CachePolicy
import logging
import mlflow

logger = logging.getLogger(__name__)


# --- You can also define custom cache policies to control caching behavior at a more granular level.
# custom_cache_policy = CachePolicy(include_step_code=False)

# @step(cache_policy=custom_cache_policy)
# def my_step():
#     ...

# # or
# my_step = my_step.with_options(cache_policy=custom_cache_policy)

@step(experiment_tracker="mlflow_tracker")
def training_data_loader() -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Load the iris dataset as a tuple of Pandas DataFrame / Series."""
    logger.info("Loading iris...")
    iris = load_iris(as_frame=True)
    logger.info("Splitting train and test...")

    # Log dataset info to MLflow
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("total_samples", len(iris.data))
    mlflow.log_metric("num_features", iris.data.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, shuffle=True, random_state=42
    )

    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))

    return X_train, X_test, y_train, y_test


@step(experiment_tracker="mlflow_tracker")
def svc_trainer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    gamma: float = 0.001,
) -> Tuple[
    Annotated[ClassifierMixin, "trained_model"],
    Annotated[float, "training_acc"],
]:
    """Train a sklearn SVC classifier."""

    # Log hyperparameters
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("model_type", "SVC")

    model = SVC(gamma=gamma)
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    train_acc = model.score(X_train.to_numpy(), y_train.to_numpy())
    print(f"Train accuracy: {train_acc}")

    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, name="svc_model", serialization_format="skops")

    return model, train_acc


@pipeline(model=Model(name="svc_iris_classifier", version=None))
def training_pipeline_mlflow(gamma: float = 0.002):
    X_train, X_test, y_train, y_test = training_data_loader()
    svc_trainer(gamma=gamma, X_train=X_train, y_train=y_train)


if __name__ == "__main__":
    # # Generate the template first
    # training_pipeline_mlflow.write_run_configuration_template(path='configs/template.yaml')
 
    training_pipeline_mlflow = training_pipeline_mlflow.with_options(
    config_path='./configs/training_config.yaml',
    enable_cache=True #override cache settings for this pipeline at runtime
    )

    training_pipeline_mlflow()