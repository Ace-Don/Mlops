# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2023. All rights reserved.
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

from typing import Any

import mlflow
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_predict(
    model: Any,
    dataset_inf: pd.DataFrame,
) -> Annotated[pd.Series, "predictions"]:
    """Predictions step.

    This is an example of a predictions step that takes the data and model in
    and returns predicted values.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use different input data.
    See the documentation for more information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        model: Trained model.
        dataset_inf: The inference dataset.

    Returns:
        The predictions as pandas series
    """
    # Setup MLflow experiment (tracking URI set globally in run.py)
    mlflow.set_experiment("inference")
    mlflow.start_run(run_name="inference_predictions")
    
    # Log inference info as parameters
    mlflow.log_param("inference_samples", dataset_inf.shape[0])
    mlflow.log_param("inference_features", dataset_inf.shape[1])
    
    # run prediction from memory
    predictions = model.predict(dataset_inf)
    
    # Get prediction probabilities if available
    try:
        prediction_probs = model.predict_proba(dataset_inf)
        avg_confidence = prediction_probs.max(axis=1).mean()
        mlflow.log_metric("average_prediction_confidence", avg_confidence)
    except AttributeError:
        pass  # Model doesn't have predict_proba

    predictions = pd.Series(predictions, name="predicted")
    
    # Log prediction statistics
    mlflow.log_metric("predictions_made", len(predictions))
    mlflow.log_metric("class_0_count", (predictions == 0).sum())
    mlflow.log_metric("class_1_count", (predictions == 1).sum())
    mlflow.end_run()
    return predictions