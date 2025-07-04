import logging
import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from src.evaluation import MSE, R2Score, RMSE
from zenml.client import Client
import mlflow



experiment_tracker = Client().active_stack.experiment_tracker  # activates the tracker using client


@step(experiment_tracker=experiment_tracker.name)  # initializes the experiment tracker and tracks the experiment components
def evalute_model(model:RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
                  
    ) -> Tuple[Annotated[float, "RMSE"], Annotated[float, "R2 Score"]]:
    
  try:    
      predictions = model.predict(X_test)
      mse_class=MSE()
      mse= mse_class.calculate_score(y_test, predictions)
      mlflow.log_metric("mse", mse)  # mean square error value
      
      r2_class=R2Score()
      r2_score=r2_class.calculate_score(y_test, predictions)
      mlflow.log_metric("r2_score", r2_score)  # return r2 score value
    
      rmse_class=RMSE()
      rmse=rmse_class.calculate_score(y_test, predictions)
      mlflow.log_metric("rmse", rmse)  # rmse return value
      
      return rmse, r2_score
  
  except Exception as e:
    logging.error(f"Error in evaluating model: {e}")
    raise e