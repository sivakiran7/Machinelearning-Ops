from pipelines.deployment_pipeline import continous_deployment_pipeline
from pipelines.deployment_pipeline import inference_pipeline
import click
from rich import print
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from typing import cast
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from pipelines.deployment_pipeline import continous_deployment_pipeline


DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "--c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT],
                      case_sensitive=False),
    default=DEPLOY_AND_PREDICT,
    help="optionally choose to only run the deployment"
    "pipe line to train and deploy a model(`deploy`), or to"
    " run the inference pipeline to make predictions(`predict`), or to run both (`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required for the model to be deployed."
)
def run_deployment(config: str, min_accuracy: float):
    """
    Run the deployment pipeline or inference pipeline based on the provided configuration.
    """
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    if deploy:
        continous_deployment_pipeline(
            data_path="D:\Anonymous\Machine learning and Deep learning\Machinelearning Ops\data\olist_customers_dataset.csv",
            min_accuracy=min_accuracy,
            workers=1,
            timeout=60,
            )
    if predict:
         inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        "you can run:\n"
        f"mlflow ui --backend-store-uri {get_tracking_uri()} to view the MLflow UI.\n"
        f"MLFlow Model Deployer: {mlflow_model_deployer_component.name}\n"
        
        "mlflow_example_pipeline experiment ther you'll also be able to compaer two are more runs.\n\n"
    )

  # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )

if __name__ =="__main__":
    run_deployment()