import json
import numpy as np
import pandas as pd
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from pydantic import BaseModel
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evalute_model

from pipelines.utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=["mlflow"])

# making an deploying a trigger for triggering the deployment of the model

class DeploymentTriggerConfig(BaseModel):
    """
    Configuration for the deployment trigger.
    """
    min_accuracy: float = 0.92
    
@step 
def dynamic_importer()->str:
    data=get_data_for_test()
    return data
@step
def deployment_trigger(
    accuracy: float ,
    config: DeploymentTriggerConfig = DeploymentTriggerConfig()
 ):
    return accuracy>=config.min_accuracy


# making an prediction 
@step(enable_cache=False)
def predicton_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running:bool = True,
    model_name: str = "model",
)->MLFlowDeploymentService:


    """
    Loads the prediction service for the model.
    """
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    
    
    print(services)
    print(type(services))
    print(services[0])
    
    # predicting the model using the service and the data
@step
def predictor(
    service: MLFlowDeploymentService,
    data:str,
)->np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

    
# creating a pipeline in continous deployment
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continous_deployment_pipeline(
    data_path:str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df= ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evalute_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data=dynamic_importer()
    service=predicton_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
        
    )
    predictor(service=service,data=data,)
    