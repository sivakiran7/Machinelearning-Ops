import logging
from abc import ABC, abstractmethod # type: ignore
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


class Model(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model
        """
        pass


class LinearRegressionModel(Model):

    def train(self, X_train, y_train, **kwargs):
        """
        Train the linear regression model
        """

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained successfully")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
