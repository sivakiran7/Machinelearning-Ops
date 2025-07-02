import logging
import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from src.evaluation import MSE, R2Score, RMSE


@step
def evalute_model(model:RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
                  
    ) -> Tuple[Annotated[float, "RMSE"], Annotated[float, "R2 Score"]]:
    
  try:    
            predictions = model.predict(X_test)
            mse_class=MSE()
            mse= mse_class.calculate_score(y_test, predictions)
            
            r2_class=R2Score()
            r2_score=r2_class.calculate_score(y_test, predictions)
            
            rmse_class=RMSE()
            rmse=rmse_class.calculate_score(y_test, predictions)
            
            return rmse, r2_score
  
  except Exception as e:
    logging.error(f"Error in evaluating model: {e}")
    raise e