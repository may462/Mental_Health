import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import shap
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from models.matrice.matrice import evaluate_model
from dataset_ml.data_ml import read_train_df2
from models.LogisticRegression.logistic_regression import train_and_evaluate_logistic_regression
from models.DecisionTree.decision_tree import train_and_evaluate_decision_tree
from models.RandomForest.random_forest import train_and_evaluate_random_forest
from models.knn.knn import train_and_evaluate_knn
from models.XGBoost.XGBoost import train_and_evaluate_xgboost
from models.Boosting.boosting import train_and_evaluate_boosting

import matplotlib
matplotlib.use('Agg')  # Usa un backend non interattivo

# Lettura Dataset
data_path = os.path.join(project_root, 'dataset_ml', 'survey2.csv')
train_df2 = read_train_df2(data_path)

# Imposta il seed per la riproducibilità
seed = 123
np.random.seed(seed)
# Salva lo stato del generatore di numeri casuali
state = np.random.get_state()

# Separa le caratteristiche dalla variabile target 
X = train_df2.drop(columns=['treatment'])
Y = train_df2['treatment']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=seed)

# Logistic Regression
train_and_evaluate_logistic_regression(x_train, x_test, y_train, y_test, seed)
# Ripristina lo stato del generatore di numeri casuali
np.random.set_state(state)

# Decision Tree
train_and_evaluate_decision_tree(x_train, x_test, y_train, y_test, seed)
# Ripristina lo stato del generatore di numeri casuali
np.random.set_state(state)

# Random Forest
train_and_evaluate_random_forest(x_train, x_test, y_train, y_test,seed)
# Ripristina lo stato del generatore di numeri casuali
np.random.set_state(state)

# KNN
train_and_evaluate_knn(x_train, x_test, y_train, y_test)
# Ripristina lo stato del generatore di numeri casuali
np.random.set_state(state)

# XGBoost
train_and_evaluate_xgboost(x_train, x_test, y_train, y_test, seed)
# Ripristina lo stato del generatore di numeri casuali
np.random.set_state(state)

# Boosting
train_and_evaluate_boosting(x_train, x_test, y_train, y_test, seed)
# Ripristina lo stato del generatore di numeri casuali
np.random.set_state(state)

