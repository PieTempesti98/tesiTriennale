import matplotlib.pyplot as plt
import seaborn as sn
from pandas import DataFrame
from numpy import mean, var


def printResults(scoreList, confusionMatrix, model, conf):
    # stampa di valor medio, varianza, valore massimo e minimo dei test svolti
    print('Score medio: {}\nDeviazione standard: {}\nMiglior risultato: {}\nPeggior risultato: {}'.format(
        mean(scoreList), var(scoreList), max(scoreList),
        min(scoreList))
    )
    # plotting della matrice di confusione
    casestudy = ['HISTORY - TOURISM', 'LEVENT', 'MASLAK - INDUSTRIES', 'TAKSIM - SHOPPING', 'UNIVERSITY']
    df_cm = DataFrame(confusionMatrix, index=casestudy, columns=casestudy)
    plt.figure(figsize=(15, 10))
    sn.heatmap(df_cm, annot=True, cbar_kws={"orientation": "horizontal"},
               annot_kws={"fontsize": 16}, cmap='Blues', fmt='g')
    titolo = "Matrice di confusione del modello " + model + " con configurazione " + conf
    plt.title(titolo, fontweight='bold', fontsize=15)
    plt.xlabel("Actual Class")
    plt.ylabel("Predicted Class")
    plt.show()
