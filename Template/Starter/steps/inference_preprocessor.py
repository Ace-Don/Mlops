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

import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step


@step
def inference_preprocessor(
    dataset_inf: pd.DataFrame,
    preprocess_pipeline: Pipeline,
    target: str,
) -> Annotated[pd.DataFrame, "inference_dataset"]:
    """Data preprocessor step.

    This is an example of a data processor step that prepares the data so that
    it is suitable for model inference. It takes in a dataset as an input step
    artifact and performs any necessary preprocessing steps based on pretrained
    preprocessing pipeline.

    Args:
        dataset_inf: The inference dataset.
        preprocess_pipeline: Pretrained `Pipeline` to process dataset.
        target: Name of target columns in dataset.

    Returns:
        The processed dataframe: dataset_inf.
    """
    # Setup MLflow experiment (tracking URI set globally in run.py)
    mlflow.set_experiment("inference")
    mlflow.start_run(run_name="inference_data_preprocessing")
    
    # Log inference dataset info as parameters
    mlflow.log_param("target_column", target)
    mlflow.log_param("inference_samples", dataset_inf.shape[0])
    mlflow.log_param("inference_features", dataset_inf.shape[1])
    
    # artificially adding `target` column to avoid Pipeline issues
    dataset_inf[target] = pd.Series([1] * dataset_inf.shape[0])
    dataset_inf = preprocess_pipeline.transform(dataset_inf)
    dataset_inf.drop(columns=[target], inplace=True)
    
    mlflow.log_metric("preprocessed_features", dataset_inf.shape[1])
    mlflow.end_run()
    return dataset_inf
