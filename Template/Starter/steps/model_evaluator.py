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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd
from sklearn.base import ClassifierMixin

from zenml import log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_evaluator(
        model: ClassifierMixin,
        dataset_trn: pd.DataFrame,
        dataset_tst: pd.DataFrame,
        min_train_accuracy: float = 0.0,
        min_test_accuracy: float = 0.0,
        target: Optional[str] = "target",
) -> float:
    """Evaluate a trained model.

    This is an example of a model evaluation step that takes in a model artifact
    previously trained by another step in your pipeline, and a training
    and validation data set pair which it uses to evaluate the model's
    performance. The model metrics are then returned as step output artifacts
    (in this case, the model accuracy on the train and test set).

    The suggested step implementation also outputs some warnings if the model
    performance does not meet some minimum criteria. This is just an example of
    how you can use steps to monitor your model performance and alert you if
    something goes wrong. As an alternative, you can raise an exception in the
    step to force the pipeline run to fail early and all subsequent steps to
    be skipped.

    This step is parameterized to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to use different values for the acceptable model performance thresholds and
    to control whether the pipeline run should fail if the model performance
    does not meet the minimum criteria. See the documentation for more
    information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        model: The pre-trained model artifact.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        min_train_accuracy: Minimal acceptable training accuracy value.
        min_test_accuracy: Minimal acceptable testing accuracy value.
        target: Name of target column in dataset.

    Returns:
        The model accuracy on the test set.
    """
    # Setup MLflow experiment (tracking URI set globally in run.py)
    mlflow.set_experiment("model_training")
    mlflow.start_run(run_name="model_evaluation")
    
    # Log evaluation parameters
    mlflow.log_param("min_train_accuracy", min_train_accuracy)
    mlflow.log_param("min_test_accuracy", min_test_accuracy)
    mlflow.log_param("target", target)
    
    # Calculate the model accuracy on the train and test set
    X_trn = dataset_trn.drop(columns=[target])
    y_trn = dataset_trn[target]
    X_tst = dataset_tst.drop(columns=[target])
    y_tst = dataset_tst[target]
    
    # Get predictions
    y_trn_pred = model.predict(X_trn)
    y_tst_pred = model.predict(X_tst)
    
    # Calculate accuracy and additional metrics
    trn_acc = accuracy_score(y_trn, y_trn_pred)
    tst_acc = accuracy_score(y_tst, y_tst_pred)
    
    # Calculate additional metrics for test set
    tst_precision = precision_score(y_tst, y_tst_pred, average="weighted", zero_division=0)
    tst_recall = recall_score(y_tst, y_tst_pred, average="weighted", zero_division=0)
    tst_f1 = f1_score(y_tst, y_tst_pred, average="weighted", zero_division=0)
    
    logger.info(f"Train accuracy={trn_acc * 100:.2f}%")
    logger.info(f"Test accuracy={tst_acc * 100:.2f}%")
    logger.info(f"Test precision={tst_precision * 100:.2f}%")
    logger.info(f"Test recall={tst_recall * 100:.2f}%")
    logger.info(f"Test F1-Score={tst_f1 * 100:.2f}%")
    
    # Log metrics to MLflow
    mlflow.log_metric("train_accuracy", trn_acc)
    mlflow.log_metric("test_accuracy", tst_acc)
    mlflow.log_metric("test_precision", tst_precision)
    mlflow.log_metric("test_recall", tst_recall)
    mlflow.log_metric("test_f1_score", tst_f1)

    messages = []
    if trn_acc < min_train_accuracy:
        messages.append(
            f"Train accuracy {trn_acc * 100:.2f}% is below {min_train_accuracy * 100:.2f}% !"
        )
    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test accuracy {tst_acc * 100:.2f}% is below {min_test_accuracy * 100:.2f}% !"
        )
    else:
        for message in messages:
            logger.warning(message)
    
    # Log warnings to MLflow
    if messages:
        mlflow.log_param("warnings", "; ".join(messages))

    mlflow.end_run()
    
    client = Client()
    latest_classifier = client.get_artifact_version("sklearn_classifier")

    log_metadata(
        metadata={
            "train_accuracy": float(trn_acc),
            "test_accuracy": float(tst_acc)
        },
        artifact_version_id=latest_classifier.id
    )

    return float(tst_acc)
