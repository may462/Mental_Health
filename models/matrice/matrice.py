import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Usa un backend non interattivo

import seaborn as sns
from sklearn import metrics

def evaluate_model(y_test, y_pred_class, save_path=None):
   
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # Chiudi tutte le figure aperte
    plt.close('all')

    # Crea la figura per la matrice di confusione
    plt.figure(figsize=(10, 7))

    # visualize Confusion Matrix
    sns.heatmap(confusion, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Salva l'immagine se il percorso è specificato
   
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Matrice di confusione salvata in: {save_path}')
    
    plt.show()

    # Calcola le metriche di valutazione
    accuracy = metrics.accuracy_score(y_test, y_pred_class)
    classification_error = 1 - accuracy
    false_positive_rate = FP / float(TN + FP)
    precision = metrics.precision_score(y_test, y_pred_class)

    # Stampa le metriche di valutazione
    print('Classification Accuracy:', accuracy)
    print('Classification Error:', classification_error)
    print('False Positive Rate:', false_positive_rate)
    print('Precision:', precision)

    metrics_dict = {
        'confusion_matrix': confusion,
        'accuracy': accuracy,
        'classification_error': classification_error,
        'false_positive_rate': false_positive_rate,
        'precision': precision
    }

    return metrics_dict
