import mlflow
from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

# Minimum test accuracy required before a model can be promoted to production.
# Set conservatively low so any trained model is eligible on first run.
PROMOTION_ACCURACY_THRESHOLD = 0.80


@step
def fraud_model_promoter(accuracy: float, stage: str = "production") -> bool:
    """Fraud model promotion step.

    Decides whether to promote the newly trained model to the given stage
    (default: 'production').

    Promotion logic:
      1. If accuracy < PROMOTION_ACCURACY_THRESHOLD (80%): reject immediately.
      2. If no model currently exists in the target stage: promote unconditionally.
      3. If a model already exists in the target stage: compare test_accuracy
         stored in its run_metadata.  Promote only if the new model is better.

    The comparison uses the 'test_accuracy' key written to artifact run_metadata
    by fraud_model_evaluator via log_metadata().

    MLflow tracking experiment: 'fraud_detection_model_training'

    Args:
        accuracy: Test accuracy of the current model (from fraud_model_evaluator).
        stage: ZenML model stage to promote to. Default 'production'.

    Returns:
        True if the model was promoted, False otherwise.
    """
    mlflow.set_experiment("fraud_detection_model_training")
    with mlflow.start_run(run_name="fraud_model_promotion_decision"):

        mlflow.log_param("fraud_candidate_accuracy", accuracy)
        mlflow.log_param("fraud_promotion_stage", stage)
        mlflow.log_param("fraud_accuracy_threshold", PROMOTION_ACCURACY_THRESHOLD)

        is_promoted = False

        # ── Gate: minimum accuracy ────────────────────────────────────────────
        if accuracy < PROMOTION_ACCURACY_THRESHOLD:
            logger.info(
                f"Model accuracy {accuracy*100:.2f}% is below the "
                f"{PROMOTION_ACCURACY_THRESHOLD*100:.0f}% threshold. Not promoting."
            )
            mlflow.log_param("fraud_promotion_status", "rejected_below_threshold")
            mlflow.end_run()
            return False

        # ── Retrieve the current model context ────────────────────────────────
        current_model = get_step_context().model
        client = Client()

        try:
            # Check if a production model already exists
            stage_model = client.get_model_version(current_model.name, stage)

            prod_accuracy = float(
                stage_model.get_artifact("fraud_trained_model")
                .run_metadata["test_accuracy"]
            )
            mlflow.log_metric("fraud_production_accuracy", prod_accuracy)

            if accuracy > prod_accuracy:
                logger.info(
                    f"New model ({accuracy*100:.2f}%) beats production "
                    f"({prod_accuracy*100:.2f}%). Promoting to '{stage}'."
                )
                current_model.set_stage(stage, force=True)
                is_promoted = True
                mlflow.log_param("fraud_promotion_status", "promoted_better_accuracy")
            else:
                logger.info(
                    f"New model ({accuracy*100:.2f}%) does not beat production "
                    f"({prod_accuracy*100:.2f}%). Keeping existing model."
                )
                mlflow.log_param("fraud_promotion_status", "rejected_production_better")

        except KeyError:
            # No model in the target stage — promote unconditionally
            logger.info(
                f"No existing model in stage '{stage}'. "
                f"Promoting first model ({accuracy*100:.2f}%)."
            )
            current_model.set_stage(stage, force=True)
            is_promoted = True
            mlflow.log_param("fraud_promotion_status", "promoted_first_model")

        mlflow.log_param("fraud_was_promoted", str(is_promoted))

    return is_promoted
