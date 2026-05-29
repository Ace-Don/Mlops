# Financial Fraud Detection (MLOps Pipeline)

This project leverages **ZenML** and **MLflow** to create a production-ready Machine Learning pipeline for detecting financial fraud. The pipeline is built upon the BankSim dataset (`bs140513_032310.csv`).

## Project Overview

Financial fraud detection requires processing large volumes of transaction data, engineering temporal features (like rolling transaction counts), and rigorously tracking model performance metrics such as Precision, Recall, and F1-Score to deal with severe class imbalances.

This repository takes a monolithic Jupyter Notebook approach and modularizes it into a scalable MLOps pipeline.

## MLOps Architecture

The pipeline consists of modular steps connected via ZenML:

### 1. Feature Engineering Pipeline (`python run.py --feature-pipeline`)
- **`data_loader`**: Reads the CSV dataset and strictly validates the schema to catch any data drift or unexpected nulls early in the lifecycle.
- **`feature_engineer`**: Performs complex temporal aggregations. It computes `1-day`, `7-day`, and `30-day` rolling transaction counts for each customer. This step processes the raw data chronologically.
- **`data_splitter`**: Splits the engineered data into training and testing sets to prevent data leakage during scaling and encoding.
- **`data_preprocessor`**: Fits an `OrdinalEncoder` on categorical columns (`age`, `gender`, `category`) and scales numeric features, ensuring transformations applied to the test set are strictly based on the training distribution.

### 2. Training Pipeline (`python run.py --training-pipeline`)
By default, the training pipeline loops over 4 distinct model configurations to find the best performer:
- **Logistic Regression (LR)**
- **XGBoost (XGB)**
- **CatBoost**
- **LightGBM**

The steps include:
- **`model_trainer`**: Trains the requested algorithm and logs model hyperparameters to MLflow.
- **`model_evaluator`**: Calculates performance metrics, specifically weighted towards fraud detection (`average="binary"`): Accuracy, Precision, Recall, and F1-Score.
- **`model_promoter`**: Consults the MLflow registry. If the newly trained model has a higher Test Accuracy (or passes the base threshold), it is automatically promoted to the `production` stage.

## Configurations

The `configs/` directory contains YAML files for each model (e.g., `training_xgb.yaml`). You can use these files to adjust:
- `model_type`: Which algorithm to use.
- `enable_cache`: ZenML caching allows skipping steps that have already been computed with the exact same inputs and parameters.

## Getting Started

### 1. Requirements
Ensure you have installed the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Pipelines
Run the pipelines using the main entry point:
```bash
# Run data loading and preprocessing
python run.py --feature-pipeline

# Run the model training loop across LR, XGBoost, CatBoost, and LightGBM
python run.py --training-pipeline
```

### 3. Tracking Experiments (MLflow)
To visualize the metrics (Precision, Recall, F1-Score) and compare the 4 models:
```bash
mlflow ui --port 5000
```
Then navigate to `http://localhost:5000` to view the runs.
