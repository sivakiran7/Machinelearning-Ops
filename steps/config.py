
from pydantic import BaseModel
class ModelNameConfig(BaseModel):
    """
    Configuration for the model name.
    """
    model_name: str = "linearRegression"
    fine_tuning: bool = False