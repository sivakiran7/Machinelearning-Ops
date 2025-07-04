from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":

    print(Client().active_stack.experiment_tracker.get_tracking_uri())  # it is used to print the active tracker used for the stack
    train_pipeline(
        data_path="D:\Anonymous\Machine learning and Deep learning\Machinelearning Ops\data\olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:C:\Users\sivak\AppData\Roaming\zenml\local_stores\889ce403-a862-41ce-953b-c465bf0d7d41\mlruns"  getting the info of the tracker used and the path of the mlflow 