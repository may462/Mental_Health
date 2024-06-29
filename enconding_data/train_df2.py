import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd

# Caricare il dataset
data_path = 'dataset/survey.csv'
train_df = pd.read_csv(data_path)

# Funzione importata da un altro modulo 
from missing_data.delete_column_NaN import train_df_delete_column

timestamp='Timestamp'
state='state'
comments='comments'

train_df2 = train_df_delete_column(train_df, 'Timestamp', 'state', 'comments')

# Creazione liste per ogni tipo di dato
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

# Assegna valori di Default per ogni tipo di dato
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

from missing_data.clean_column import clean_and_process_data

# Pulire e processare il dataset
train_df2, missing_data = clean_and_process_data(train_df2, intFeatures, stringFeatures, floatFeatures, defaultInt, defaultString, defaultFloat)


from enconding_data.enc_data import enconding
train_df2 = enconding(train_df2)

# Visualizzare i risultati
print("\n",train_df2)

train_df2.to_csv('dataset_ml/survey2.csv', index=False)