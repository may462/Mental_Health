import sys
import os

# Aggiungi la directory principale del progetto al percorso di ricerca
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFE
import seaborn as sns
from models.matrice.matrice import evaluate_model
from dataset_ml.data_ml import read_train_df2

# Lettura Dataset
data_path = os.path.join(project_root, 'dataset_ml', 'survey2.csv')
train_df2 = read_train_df2(data_path)

# Funzione da richiamare nel Main
def train_and_evaluate_xgboost(x_train, x_test, y_train, y_test, seed):

    # XGBoost
    model = XGBClassifier(random_state=seed)
    model.fit(x_train, y_train)


    # Matrice di confusione 
    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(x_test)

    # Valuta il modello e salva la matrice di confusione in formato PNG
    save_path = os.path.join(project_root, 'results', 'XGBoost', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    metrics_results = evaluate_model(y_test, y_pred, save_path=save_path)

    # Report di classificazione
    print("\n","REPORT DI CLASSIFICAZIONE XGBOOST", "\n")
    print(classification_report(y_test, y_pred))



    # Metodi di Feature Selection
    feature_selection_methods = {
        "Chi-Squared": SelectKBest(score_func=chi2, k=5),  
        "Forward Selection": SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=5, direction='forward'),
        "Backward Selection": SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=5, direction='backward'),
        "Exhaustive Search": SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select=5),
        "ANOVA F-value": SelectKBest(score_func=f_classif, k=5),
        "Mutual Information": SelectKBest(score_func=mutual_info_classif, k=5),
        "Tree-based Feature Selection": SelectFromModel(RandomForestClassifier(n_estimators=100))
    }

    # Addestramento e valutazione dei modelli con i metodi di Feature Selection
    for method, selector in feature_selection_methods.items():

        # Applica il metodo di Feature Selection
        X_train_selected = selector.fit_transform(x_train, y_train)
        X_test_selected = selector.transform(x_test)
        
        # Addestra un classificatore
        XGBoost = XGBClassifier(random_state=seed)  
        XGBoost.fit(X_train_selected, y_train)

        # Ottieni gli indici delle feature selezionate
        selected_indices = selector.get_support(indices=True)
        
        # Ottieni i nomi delle feature selezionate
        selected_features = train_df2.columns[selected_indices]

        # Valuta il modello
        y_pred = XGBoost.predict(X_test_selected)
        print(f"\nREPORT DI CLASSIFICAZIONE CON  {method}:\n")
        print(classification_report(y_test, y_pred),"\n")
        print(f"FEATURE SELEZIONATE CON {method}: {selected_features.tolist()} \n")


    # SHAP
    def f(x):
        return model.predict_proba(x)[:, 1]
    med = x_train.median().values.reshape((1, x_train.shape[1]))
    explainer = shap.Explainer(f, med)
    shap_values = explainer(x_test.iloc[0:1000, :])

    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False)


    # Salva il grafico SHAP in formato PNG
    shap_path = os.path.join(project_root, 'results', 'XGBoost', 'shap.png')
    os.makedirs(os.path.dirname(shap_path), exist_ok=True)
    plt.savefig(shap_path, format='png')
    plt.close()


    # Seleziona le prime 5 feature più importanti
    shap_values_array = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': x_train.columns,
        'importance': shap_values_array
    })
    important_features = importance_df.sort_values(by='importance', ascending=False).head(5)
    important_feature_names = important_features['feature'].tolist()

    # Filtra il dataset per mantenere solo le 5 feature più importanti
    x_train_selected = x_train[important_feature_names]
    x_test_selected = x_test[important_feature_names]

    # Allena nuovamente il modello con le feature selezionate
    model_selected = XGBClassifier(random_state=seed)
    model_selected.fit(x_train_selected, y_train)


    y_pred_selected = model_selected.predict(x_test_selected)
    print("\n", "REPORT DI CLASSIFICAZIONE CON SHAP:", "\n")
    print(classification_report(y_test, y_pred_selected))

    print("FEATURE SELEZIONATE CON SHAP:", important_feature_names)

    