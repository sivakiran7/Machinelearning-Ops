from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluation import evalute_model
from steps.model_train import train_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    # Assuming evalute_model takes the model and test data as input
    
    r2_score, rmse = evalute_model(model, X_test, y_test)
    
    
