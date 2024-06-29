import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd

from missing_data.delete_column_NaN import train_df_delete_column
from dataset.data import read_train_df

data_path='dataset/survey.csv'
train_df=read_train_df(data_path)


timestamp='Timestamp'
state='state'
comments='comments'


train_df2=train_df_delete_column(train_df,timestamp,state,comments)


# Pulire valori NaN

def clean_and_process_data(train_df2, intFeatures, stringFeatures, floatFeatures, defaultInt, defaultString, defaultFloat):
    for feature in train_df2:
        if feature in intFeatures:
            train_df2[feature] = train_df2[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            train_df2[feature] = train_df2[feature].fillna(defaultString)
        elif feature in floatFeatures:
            train_df2[feature] = train_df2[feature].fillna(defaultFloat)
        else:
            print('Error: Feature %s not recognized.' % feature)


    # Pulire feature 'Gender'

    # Converte tutti i valori in minuscolo
    gender = train_df2['Gender'].str.lower()
    

    # Seleziona elementi unici
    gender = train_df2['Gender'].unique()

    # Tre gruppi di genere
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
    trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
    female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

    for (row, col) in train_df2.iterrows():

        if str.lower(col.Gender) in male_str:
            train_df2['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

        if str.lower(col.Gender) in female_str:
            train_df2['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

        if str.lower(col.Gender) in trans_str:
            train_df2['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

    # Rimuove da Gender valori di stk_list
    stk_list = ['A little about you', 'p']
    train_df2 = train_df2[~train_df2['Gender'].isin(stk_list)]

    print("\nGENDER\n",train_df2['Gender'].unique(),"\n")

    # Completa i dati NaN di Age con la media
    train_df2.loc[:, 'Age'] = train_df2['Age'].fillna(train_df2['Age'].median())


    # Limita valori Age < 18 and > 120
    train_df2.loc[:,'Age'] = train_df2['Age'].clip(lower=18, upper=120)

    # Sostituisce valori NaN di self employed con No 
    train_df2.loc[:,'self_employed'] = train_df2['self_employed'].replace([defaultString], 'No')
    print("\nSELF EMPLOYED\n",train_df2['self_employed'].unique(),"\n")

    # Sostituisce valori NaN di work interfere con Non lo so 

    train_df2.loc[:,'work_interfere'] = train_df2['work_interfere'].replace([defaultString], 'Don\'t know' )
    print("\nWORK INTERFERE\n",train_df2['work_interfere'].unique(),"\n")


    #controllo valori NaN
   
    total = train_df2.isnull().sum().sort_values(ascending=False)
    percent = (train_df2.isnull().sum()/train_df2.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)

    return train_df2, missing_data
