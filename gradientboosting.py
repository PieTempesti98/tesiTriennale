from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


class GBClassifier:
    def __init__(self, path):
        self.data = read_csv(path)
        self.confusion_matrix = np.zeros((5, 5))
        self.scoreList = []

    def training_testing(self, rep, settings):
        # split del dataframe in training e testing
        x = self.data.iloc[:, 1:]
        y = self.data.iloc[:, 0]
        train_data, test_data, train_label, test_label = train_test_split(x, y)

        # addestramento del modello Gradient Boosting a seconda del tipo di richiesta: default o tuned
        if settings == 'default':
            modello = GradientBoostingClassifier()

        else:
            modello = GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.5,
                        max_depth=2,
                        min_samples_leaf=3,
                        random_state=0,
                        max_features='sqrt'
                        )
        modello.fit(train_data, train_label)

        # calcolo lo score e lo salvo in un array per il calcolo delle statisitche finali
        prediction = modello.predict(test_data)
        test_score = accuracy_score(test_label, prediction)
        testo = "\n Test " + str(rep) + "; score: " + str(test_score)
        path = "results/GradientBoosting/scores.txt"
        file = open(path, "a")
        file.write(testo)
        self.scoreList.append(test_score)

        cm = confusion_matrix(test_label, prediction)
        self.confusion_matrix = np.add(self.confusion_matrix, cm)

    def grid_search(self, rep, params):
        # split del dataframe in training e testing
        x = self.data.iloc[:, 1:]
        y = self.data.iloc[:, 0]
        train_data, test_data, train_label, test_label = train_test_split(x, y)
        # effettuo la grid search e stampo i parametri selezionati dalla ricerca
        gsearch = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=params)
        gsearch.fit(train_data, train_label)
        print(str(gsearch.best_params_))
        print(str(gsearch.best_score_))
