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

# Visualizzare i risultati
print("\nGENDER\n", train_df2['Gender'].unique(), "\n")
print("\nSELF EMPLOYED\n", train_df2['self_employed'].unique(), "\n")
print("\nWORK INTERFERE\n", train_df2['work_interfere'].unique(), "\n")
print(train_df2)
print("\nCONTROLLO VALORI NAN\n", missing_data)


from sklearn import preprocessing
from sklearn.preprocessing import  LabelEncoder

# Encoding Data
def enconding (train_df2):
    labelDict = {}
    for feature in train_df2 :
        le = preprocessing.LabelEncoder()
        le.fit(train_df2[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        train_df2[feature] = le.transform(train_df2[feature])
        if train_df2[feature].name == 'Age' and train_df2[feature].dtype != 'object':
            train_df2[feature] = le.inverse_transform(train_df2[feature])

        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] =labelValue

        # Mappatura tra le categorie originali e le etichette codificate
        print(f"\nFeature: {feature}\n")
        for category, encoded_label in zip(le.classes_, le_name_mapping.values()):
            print(f"{category} -> {encoded_label}\n")

    for key, value in labelDict.items():
        print(key, value)


    return train_df2


