import pandas as pd

# Funzione che legge il Dataset in formato processabile con il ML
def read_train_df2(data_path):

    train_df2 = pd.read_csv(data_path)

    return train_df2