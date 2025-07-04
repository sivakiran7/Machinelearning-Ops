import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


# evaluation of logetsic regression model using evalution metrics
class Evaluation(ABC):
    """
    Abstract class for evaluation strategies.
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model on the test data.
        """
        pass


class MSE(Evaluation):
    """
    Mean Squared Error evaluation strategy.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error.
        """
        try:
            logging.info("Calculating Mean Squared Error (MSE)...")
            mse = np.mean((y_true - y_pred) ** 2)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e


class R2Score(Evaluation):
    """
    evaluation strategy for R2 Score.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating R2 Score...")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e


class RMSE(Evaluation):

    def calculate_score(self, y_true, y_pred):
        try:
            logging.info("calculating Root Mean Squared Error (RMSE)...")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e
