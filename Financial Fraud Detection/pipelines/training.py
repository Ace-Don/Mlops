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
from uuid import UUID

from steps import model_evaluator, model_promoter, model_trainer
from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

from pipelines import (
    feature_engineering,
)
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps import evidently_report_step

logger = get_logger(__name__)


@pipeline
def training(
    train_dataset_id: Optional[UUID] = None,
    test_dataset_id: Optional[UUID] = None,
    target: Optional[str] = "target",
    model_type: Optional[str] = "sgd",
):
    """
    Model training pipeline.

    This is a pipeline that loads the data from a preprocessing pipeline,
    trains a model on it and evaluates the model. If it is the first model
    to be trained, it will be promoted to production. If not, it will be
    promoted only if it has a higher accuracy than the current production
    model version.

    Args:
        train_dataset_id: ID of the train dataset produced by feature engineering.
        test_dataset_id: ID of the test dataset produced by feature engineering.
        target: Name of target column in dataset.
        model_type: The type of model to train.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.

    # Execute Feature Engineering Pipeline
    if train_dataset_id is None or test_dataset_id is None:
        dataset_trn, dataset_tst = feature_engineering()
    else:
        client = Client()
        dataset_trn = client.get_artifact_version(name_id_or_prefix=train_dataset_id)
        dataset_tst = client.get_artifact_version(name_id_or_prefix=test_dataset_id)

    # Calculate Data Drift
    evidently_report_step.with_options(
        name="data_drift_detector",
        parameters=dict(
            metrics=[EvidentlyMetricConfig.metric("DataDriftPreset")]
        )
    )(
        reference_dataset=dataset_trn,
        comparison_dataset=dataset_tst,
    )

    model = model_trainer(dataset_trn=dataset_trn, target=target, model_type=model_type)

    acc = model_evaluator(
        model=model,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
    )

    model_promoter(accuracy=acc)
