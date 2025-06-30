from zenml import pipelines
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluation import evalute_model
from steps.model_train import train_model

@pipelines()
def training_pipeline(data_path: str):
    