import os
from typing import Optional
from uuid import UUID

import click
import mlflow
import yaml
from pipelines import (
    fraud_feature_engineering_pipeline,
    fraud_training_pipeline,
)
from zenml.client import Client
from zenml.logger import get_logger

# Set MLflow tracking URI globally
mlflow.set_tracking_uri("http://localhost:5000")

logger = get_logger(__name__)


@click.command(
    help="""
Fraud Detection MLOps Project.

Run the ZenML pipelines for the fraud detection project.

Examples:

  \b
  # Run the feature engineering pipeline
    python run.py --feature-pipeline
  
  \b
  # Run the training pipeline (defaults to Logistic Regression if no --model-type passed)
    python run.py --training-pipeline

  \b 
  # Run the training pipeline with specific model type (lr, xgb, catboost, lightgbm)
    python run.py --training-pipeline --model-type=xgb

  \b 
  # Run the training pipeline using versioned artifacts from feature engineering
    python run.py --training-pipeline --train-dataset-version-name=1 --test-dataset-version-name=1

"""
)
@click.option(
    "--train-dataset-name",
    default="fraud_preprocessed_dataset_trn",
    type=click.STRING,
    help="The name of the train dataset produced by feature engineering.",
)
@click.option(
    "--train-dataset-version-name",
    default=None,
    type=click.STRING,
    help="Version of the train dataset produced by feature engineering. "
    "If not specified, a new version will be created via the feature pipeline.",
)
@click.option(
    "--test-dataset-name",
    default="fraud_preprocessed_dataset_tst",
    type=click.STRING,
    help="The name of the test dataset produced by feature engineering.",
)
@click.option(
    "--test-dataset-version-name",
    default=None,
    type=click.STRING,
    help="Version of the test dataset produced by feature engineering. "
    "If not specified, a new version will be created via the feature pipeline.",
)
@click.option(
    "--model-type",
    default="lr",
    type=click.Choice(["lr", "xgb", "catboost", "lightgbm"]),
    help="Model type to train when running the training pipeline.",
)
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--training-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)

@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(
    train_dataset_name: str,
    train_dataset_version_name: Optional[str],
    test_dataset_name: str,
    test_dataset_version_name: Optional[str],
    model_type: str,
    feature_pipeline: bool,
    training_pipeline: bool,
    no_cache: bool,
):
    """Main entry point for the pipeline execution."""
    client = Client()

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # ── Execute Feature Engineering Pipeline ──────────────────────────────────
    if feature_pipeline:
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, "feature_engineering.yaml"
        )
        
        run_args_feature = {}
        fraud_feature_engineering_pipeline.with_options(**pipeline_args)(**run_args_feature)
        logger.info("Feature Engineering pipeline finished successfully!\n")

        try:
            train_dataset_artifact = client.get_artifact_version(train_dataset_name)
            test_dataset_artifact = client.get_artifact_version(test_dataset_name)
            logger.info(
                "The latest feature engineering pipeline produced the following artifacts: \n"
                f"1. Train Dataset - Name: {train_dataset_name}, Version Name: {train_dataset_artifact.version} \n"
                f"2. Test Dataset: Name: {test_dataset_name}, Version Name: {test_dataset_artifact.version}"
            )
        except Exception as e:
            logger.warning(f"Could not fetch artifacts after feature pipeline: {e}")

    # ── Execute Training Pipeline ─────────────────────────────────────────────
    if training_pipeline:
        run_args_train = {}

        if train_dataset_version_name or test_dataset_version_name:
            if not (train_dataset_version_name and test_dataset_version_name):
                raise ValueError("Both train and test dataset versions must be specified.")
            
            train_dataset_artifact_version = client.get_artifact_version(
                train_dataset_name, train_dataset_version_name
            )
            test_dataset_artifact_version = client.get_artifact_version(
                test_dataset_name, test_dataset_version_name
            )
            
            run_args_train["train_dataset_id"] = train_dataset_artifact_version.id
            run_args_train["test_dataset_id"] = test_dataset_artifact_version.id

        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
            
        config_filename = f"training_{model_type}.yaml"
        pipeline_args["config_path"] = os.path.join(config_folder, config_filename)
        
        fraud_training_pipeline.with_options(**pipeline_args)(**run_args_train)
        logger.info(f"Training pipeline with {model_type} finished successfully!\n\n")




if __name__ == "__main__":
    main()
