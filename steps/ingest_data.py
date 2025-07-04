import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    ingest the data path
    """
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"ingesting data from{self.data_path}")
        return pd.read_csv(self.data_path)

# This code defines a step in a ZenML pipeline that ingests data from a specified path and returns it as a pandas DataFrame. It uses the `IngestData` class to handle the data ingestion process, which reads a CSV file from the provided path.
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    ingesting the data from the data_path
    
    args: data_path to the data
    return : pd.dataframe: the ingested data
    """
    try:
        ingest_data =IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"error while ingesting the data:{e}")
        raise e
