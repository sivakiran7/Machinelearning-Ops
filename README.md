# MLOps - zenml , mlflow

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


![Screenshot 2025-07-02 214556](https://github.com/user-attachments/assets/3191eb30-01e7-4a0d-9513-c6e469b0c654)

![Screenshot 2025-07-02 214539](https://github.com/user-attachments/assets/766bb7ad-b3e9-426b-bac0-240c7d3353f9)
