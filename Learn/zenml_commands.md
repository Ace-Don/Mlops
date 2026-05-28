# ZenML Commands Cheatsheet

---

## Setup

```bash
# Install ZenML
pip install zenml==0.94.2

# Install specific integration
zenml integration install mlflow -y

# Initialise ZenML in a project folder
zenml init

# Check ZenML version
zenml version

# Check current status (server, stack, user)
zenml status
```

---

## Server

```bash
# Login to a running server (Docker)
zenml login http://localhost:8080

# Run local server in blocking mode (no Docker)
zenml login --local --blocking
```

---

## Stacks

```bash
# List all stacks
zenml stack list

# Describe current active stack
zenml stack describe

# Register a new stack
zenml stack register my_stack \
  -o default \
  -a default \
  -e mlflow_tracker

# Set a stack as active
zenml stack set my_stack

# Update a stack component
zenml stack update my_stack -e mlflow_tracker   # update experiment tracker
zenml stack update my_stack -a my_artifact_store # update artifact store
zenml stack update my_stack -o my_orchestrator   # update orchestrator

# Update multiple components at once
zenml stack update my_stack \
  -e mlflow_tracker \
  -a my_artifact_store

# Copy a stack
zenml stack copy my_stack my_stack_copy

# Delete a stack
zenml stack delete my_stack
```

---

## Experiment Tracker (MLflow)

```bash
# Register MLflow as experiment tracker (local)
zenml experiment-tracker register mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri=http://127.0.0.1:5000

# Register MLflow as experiment tracker (remote — needs credentials)
zenml experiment-tracker register mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri=http://localhost:5000 \
  --tracking_username=my_username \
  --tracking_password=my_password

# List experiment trackers
zenml experiment-tracker list

# Update experiment tracker URI
zenml experiment-tracker update mlflow_tracker \
  --tracking_uri=http://127.0.0.1:5000

# Update experiment tracker credentials
zenml experiment-tracker update mlflow_tracker \
  --tracking_username=MY_USERNAME \
  --tracking_password=MY_PASSWORD \
  --tracking_token=MY_TOKEN

# Delete an experiment tracker
zenml experiment-tracker delete mlflow_tracker
```

---

## Artifact Store

```bash
# List artifact stores
zenml artifact-store list

# Register a local artifact store
zenml artifact-store register my_artifact_store \
  --flavor=local \
  --path=/path/to/store

# Update artifact store path
zenml artifact-store update my_artifact_store \
  --path=/new/path/to/store

# Delete an artifact store
zenml artifact-store delete my_artifact_store
```

---

## Orchestrator

```bash
# List orchestrators
zenml orchestrator list

# Register a local orchestrator
zenml orchestrator register my_orchestrator \
  --flavor=local

# Update an orchestrator
zenml orchestrator update my_orchestrator \
  --flavor=local

# Delete an orchestrator
zenml orchestrator delete my_orchestrator
```

---

## Pipelines

```bash
# List all pipelines
zenml pipeline list

# List all runs of a pipeline
zenml pipeline runs list

# Delete a pipeline
zenml pipeline delete my_pipeline
```

---

## Models

```bash
# List all models
zenml model list

# Describe a model
zenml model describe my_model

# Update a model
zenml model update my_model --tag new_tag

# Delete a model
zenml model delete my_model
```

---

## Projects

```bash
# List all projects
zenml project list

# Register a project
zenml project register -n my_project

# Update a project
zenml project update my_project --description "My updated project"

# Set active project
zenml project set my_project

# Delete a project
zenml project delete my_project
```

---

## Docker (for reference)

```bash
# Run ZenML server in Docker
docker run -it -d -p 8080:8080 \
  -v zenml_data:/zenml/.zenconfig \
  zenmldocker/zenml-server:0.94.2

# Run MLflow in Docker
docker run -d -p 5000:5000 \
  -v C:\Users\Nonso\mlflow-data:/mlflow \
  ghcr.io/mlflow/mlflow \
  mlflow server --host 0.0.0.0 --backend-store-uri /mlflow

# List running containers
docker ps

# Stop a container
docker stop <container_id>

# View container logs
docker logs <container_id>
```

---

## Useful Tips

- Always match pip ZenML version with Docker ZenML version
- Use `zenml stack describe` to confirm your active setup before running pipelines
- Use `zenml integration install mlflow -y` before registering MLflow tracker
- Artifacts stored locally cannot be viewed from the Docker server — use a shared store or MLflow for full visibility