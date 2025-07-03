import logging
import pandas as pd

from zenml import step
from sklearn.base import RegressorMixin
from src.model_development import LinearRegressionModel
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig= ModelNameConfig(model_name="linearregression", fine_tuning=False),

) -> RegressorMixin:

    try:
        model = None
        if config.model_name == "linearregression":
            trained_model = LinearRegressionModel().train(X_train, y_train)
            logging.info(f"Trained {config.model_name} model successfully.")
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
