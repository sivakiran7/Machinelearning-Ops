# MLOps : zenml , MLFlow ⚛
 
Aim of the project is to build and deploy an machine learning model from scratch using ZENML and MLFLOW 

for zenml use an updated version not facing an queries
## https://docs.zenml.io/deploying-zenml/deploying-zenml






## zenml commands to create an stack id and default project

provied the name of the project for creation of a default project
```md
zenml project register <project_name>

```

set the default project to the working directory 
```md
zenml project set <project_name>

```

## before making a project make sure to create an virtual environment for not getting an dependency errors

```md
python -m venv env
# activating the env
env\Scripts\activate

```
install zenml and upgrade pip if required dont use latest version of zenml for not facing an queries
```md
pip install --upgrade pip
pip install zenml
```

## Implementation of trackers in zenml 

In ZenML, trackers are used to monitor, log, and visualize metrics, artifacts, and other metadata throughout your machine learning (ML) pipeline lifecycle. They help you track experiments, compare models, and debug pipelines effectively.



| Use Case                   | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| **Metric Tracking**        | Log custom metrics like accuracy, loss, precision, etc. |
| **Artifact Tracking**      | Track data, models, and intermediate outputs.           |
| **Experiment Tracking**    | Compare pipeline runs, parameters, and results.         |
| **Model Versioning**       | Keep record of model changes over time.                 |
| **Debugging and Auditing** | Monitor what happened in each step for reproducibility. |

| Feature                | Supported in Tracker?            |
| ---------------------- | -------------------------------- |
| Log metrics            | ✅ `context.track.metric(...)`    |
| Log hyperparameters    | ✅ `context.track.parameter(...)` |
| Compare experiments    | ✅ via dashboard or CLI           |
| Integration with tools | ✅ MLflow, W\&B, etc.             |



![Screenshot 2025-07-02 214539](https://github.com/user-attachments/assets/766bb7ad-b3e9-426b-bac0-240c7d3353f9)

| Component              | Purpose                                                                                                                   | Examples                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **Stack**              | The environment where your pipeline runs. A stack consists of multiple components like orchestrator, artifact store, etc. | Local stack, cloud stack                |
| **Stack Components**   | Individual pieces that make up a stack. Each handles a specific part of the pipeline lifecycle.                           | Orchestrator, Artifact Store, etc.      |
| **Pipelines**          | A series of connected steps that define the ML workflow.                                                                  | Training pipeline, evaluation pipeline  |
| **Steps**              | Individual units of execution in a pipeline. They can handle data loading, model training, etc.                           | `load_data()`, `train_model()`          |
| **Artifacts**          | Outputs from steps that are stored and reused.                                                                            | Trained model, processed dataset        |
| **Metadata Store**     | Stores metadata about pipeline runs, artifacts, parameters, etc.                                                          | SQLite, PostgreSQL                      |
| **Experiment Tracker** | Logs metrics and parameters to track and compare experiments.                                                             | MLflow, Weights & Biases                |
| **Model Deployer**     | Manages deployment of trained models.                                                                                     | Seldon, BentoML, AWS Sagemaker          |
| **Secrets Manager**    | Manages API keys, credentials, and other secrets securely.                                                                | AWS Secrets Manager, GCP Secret Manager |
| **Container Registry** | Stores Docker images used during orchestration.                                                                           | DockerHub, AWS ECR                      |


steps for integration of mlflow into zenml 
```md
zenml integration install mlflow -y

```
Regestering  a tracker
```md
zenml experiment-tracker register <name of your tracker> --flavor=mlflow
```
Regestring a model deployer
```md
zenml model-deployer register mlflow --flavor=mlflow
```
seting a stack for deploying and managing machine model and 
```md
zenml stack register mlflow_stack_customer -a default -o default -d mlflow_customer -e mlflow_tracker --set
```



![Screenshot 2025-07-03 163334](https://github.com/user-attachments/assets/cd53591d-ca57-4f78-9d86-e5fe11292f07)

## For describling the models Orchestor, Model deployer, Artifact store
```md
zenml stack describe
```

![Screenshot 2025-07-03 163348](https://github.com/user-attachments/assets/f77a4db5-6fec-4dfa-8248-ba56fb5e541d)


![Screenshot 2025-07-03 163555](https://github.com/user-attachments/assets/72b0c93d-e8d8-4133-b68c-c44e5638a1c2)

# Architecture for the model implementation for integration and deployment of model 


![Screenshot 2025-07-03 163617](https://github.com/user-attachments/assets/fa995348-d174-4fe5-80d9-70adfc469fe1)



                ┌───────────────────────────────┐
                │        Your Codebase          │
                │ ───────────────────────────── │
                │  - ZenML Pipelines            │
                │  - Steps (train, eval, etc.)  │
                │  - Stack Configuration        │
                └────────────┬──────────────────┘
                             │
                    [Pipeline Run Trigger]
                             │
                             ▼
            ┌──────────────────────────────────────────────┐
            │                  ZenML Core                  │
            │ ──────────────────────────────────────────── │
            │  - Pipeline Orchestration Engine             │
            │  - Step Execution Manager                    │
            │  - Metadata Tracker                          │
            └────────────────┬─────────────────────────────┘
                             │
         ┌───────────────────┴────────────────────────────┐
         ▼                                                 ▼
     ┌──────────────────────┐                     ┌────────────────────────┐
     │   Stack Components    │   Interact via API │   External Systems      │
     │ (Plugins/Backends)    │◄──────────────────►│ (Cloud Services, Tools) │
     └──────────────────────┘                     └────────────────────────┘








