import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset.data import read_train_df
import pandas as pd


data_path='dataset/survey.csv'
train_df=read_train_df(data_path)

#Pandas: whats the data row count?
print("\nROW AND COLUMN")
print(train_df.shape,"\n")

#Pandas: whats the distribution of the data?
print("DISTRIBUTION OF THE DATA")
print(train_df.describe(),"\n")

#Pandas: What types of data do i have?
print("TYPES OF DATA")
print(train_df.info(),"\n")

print(train_df.head(),"\n")

