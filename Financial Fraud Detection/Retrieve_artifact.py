# from zenml.client import Client

# client = Client()
# artifact = client.get_artifact_version("dataset_trn", version="1")
# print(artifact)

from zenml.client import Client
from zenml.enums import ModelStages

client = Client()

# Get the production model version
mv = client.get_model_version("breast_cancer_classifier", ModelStages.PRODUCTION)

# Basic info
print("=" * 50)
print(f"Model Name:    {mv.model.name}")
print(f"Version Name:  {mv.name}")
print(f"Version ID:    {mv.id}")
print(f"Stage:         {mv.stage}")
print(f"Created:       {mv.created}")
print(f"Updated:       {mv.updated}")

# All metadata logged to this version
print("\n" + "=" * 50)
print("Run Metadata:")
for key, value in mv.run_metadata.items():
    print(f"  {key}: {value.value}")

# All artifacts linked to this version
print("\n" + "=" * 50)
print("Linked Artifacts:")
for name, artifact in mv.get_artifacts_versions().items():
    print(f"  {name}: {artifact}")

# Pipeline runs that produced this version
print("\n" + "=" * 50)
print("Pipeline Runs:")
for run_id in mv.pipeline_run_ids:
    print(f"  {run_id}")

# Load and inspect the actual model
print("\n" + "=" * 50)
print("Model Details:")
model = mv.get_artifact("sklearn_classifier").load()
print(f"  Type:   {type(model).__name__}")
print(f"  Params: {model.get_params()}")