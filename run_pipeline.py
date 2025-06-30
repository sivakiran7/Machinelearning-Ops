from pipelines.training_pipeline import train_pipeline

if __name__=="__main__":
    #run pipe line
    train_pipeline(data_path=".data\olist_customers_dataset.csv")