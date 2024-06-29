import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset.data import read_train_df

import pandas as pd


def train_df_delete_column(train_df,timestamp,state,comments):

# Rimuovi colonne "Timestamp",“comments”, “state” 
    train_df2 = train_df.drop([timestamp,state,comments], axis= 1)  #axis=1, colonne / axis=0 righe

    train_df2.isnull().sum().max() # Controllo valori NaN

    return train_df2