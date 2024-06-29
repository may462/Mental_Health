import pandas as pd

# Funzione che legge il Dataset
def read_train_df(data_path):

    train_df = pd.read_csv(data_path)

    return train_df


