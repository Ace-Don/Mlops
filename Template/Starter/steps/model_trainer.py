# Apache Software License 2.0
# 
# Copyright (c) ZenML GmbH 2026. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_trainer(
    dataset_trn: pd.DataFrame,
    model_type: str = "sgd",
    target: Optional[str] = "target",
) -> Annotated[
    ClassifierMixin, ArtifactConfig(name="sklearn_classifier", is_model_artifact=True)
]:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        model_type: The type of model to train.
        target: The name of the target column in the dataset.

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
    """
    # Setup MLflow experiment (tracking URI set globally in run.py)
    mlflow.set_experiment("model_training")
    mlflow.start_run(run_name=f"train_model_{model_type}")
    
    # Log model type parameter
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("target", target)
    
    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.
    if model_type == "sgd":
        model = SGDClassifier()
        mlflow.log_param("loss", "hinge")
        mlflow.log_param("penalty", "l2")
    elif model_type == "rf":
        model = RandomForestClassifier()
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
    else:
        mlflow.end_run()
        raise ValueError(f"Unknown model type {model_type}")
    logger.info(f"Training model {model}...")

    model.fit(
        dataset_trn.drop(columns=[target]),
        dataset_trn[target],
    )
    
    # Log the trained model to MLflow
    mlflow.sklearn.log_model(model, "sklearn_classifier")
    
    # Log training dataset info as parameters (data properties)
    mlflow.log_param("training_samples", len(dataset_trn))
    mlflow.log_param("training_features", len(dataset_trn.columns) - 1)
    
    mlflow.end_run()
    return model
