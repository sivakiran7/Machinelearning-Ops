import logging
import pandas as pd

from zenml import step


@step
def evalute_model(df: pd.DataFrame) -> None:
    pass
