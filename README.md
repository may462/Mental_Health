## MACHINE LEARNING FOR MENTAL HEALTH
Questa repository contiene il progetto di tesi svolto da Maria De Rosa presso l'Università degli Studi di Bari "Aldo Moro".

Il progetto analizza un dataset di sondaggi sulla salute mentale per costruire e valutare vari modelli di machine learning. L'obiettivo è prevedere se un paziente debba essere curato o meno per la sua malattia mentale in base ai valori ottenuti nel set di dati. Sarà importante, quindi, comprendere i fattori che influenzano la salute mentale e prevedere i risultati utilizzando diversi algoritmi di classificazione.

## DATASET 
Il dataset utilizzato è survey, è possibile scaricarlo al seguente link: https://www.kaggle.com/code/kairosart/machine-learning-for-mental-health-1/input

## CONFIGURAZIONE
- Posiziona il dataset "survey.csv" nella directory appropriata 
- Scarica le dipendenze con il comando: pip install -r requirements.txt

## RISOLUZIONE DEI PROBLEMI
- Verifica che survey.csv sia posizionato nella directory appropriata
- Assicurati che tutte le dipendenze siano installate come specificato in requirements.txt

## CONTRIBUTI
[@Maria De Rosa](https://github.com/may462/Mental_Health.git)


## UTILIZZO E DESCRIZIONE DELLE CARTELLE

## dataset
Questa cartella contiene 2 file:
- data.py = contiene le funzione che legge il dataset
- survey.csv (dataset)

## analysis
Contiene analysis.py che analizza il dataset

## missing_data
Contiene 3 file che gestiscono i dati mancanti
- search.py = cerca quali colonne contengono valori NaN
- delete_column_NaN.py = elimina le colonne con valori NaN
- clean_column.py = pulisce le colonne

## enconding_data
Contiene 2 file per la codifica dei dati
- enc_data.py = da etichette a valori, il dataset adesso è processabile con il ML e viene salvato in
    ## dataset_ml
    survey2.csv
- train_df2.py = contiene le funzione che legge il nuovo dataset elaborato

## models
6 modelli + Matrice + Main. Applicazione dei metodi di Feature Selection, applicazione dei classificatori e l'utilizzo di SHAP come metodo di spiegabilità e Feature Selection. Allena e valuta i seguenti modelli:
- ## Boosting
boosting.py 
- ## DecisionTree
decision_tree.py
- ## Knn
knn.py
- ## LogisticRegression
logistic_regression.py
- ## RandomForest
random_forest.py
- ## XGBoost
XGBoost.py
- ## matrice
matrice.py contiene la funzione per generare la matrice di confusione che viene richiamata in ogni modello
- ## Main
main.py. Vengono richiamati tutti i modelli

## results
contiene files .png
per ogni classificatore abbiamo 2 immagini
- confusion_matrix.png = matrice  di confusione
- shap.png = grafico shap di tipo summary plot

## requirements.txt
numpy==1.25.2
shap==0.45.1
scipy==1.11.4
scikit-learn==1.2.2
pandas==2.0.3
matplotlib==3.4
seaborn==0.13.1
slicer==0.0.8
xgboost==2.0.3
shap matplotlib
python==3.10
ipython==7.34.0
matplotlib-inline==0.1.7

