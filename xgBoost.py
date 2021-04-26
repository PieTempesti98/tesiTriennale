from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


class XGBoostClassifier:
    def __init__(self, path):
        self.data = read_csv(path)
        self.confusion_matrix = np.zeros((5, 5))
        self.scoreList = []

    def training_testing(self, rep, settings):
        # split del dataframe in training e testing
        x = self.data.iloc[:, 1:]
        y = self.data.iloc[:, 0]
        train_data, test_data, train_label, test_label = train_test_split(x, y)

        if settings == 'default':
            modello = XGBClassifier()
        else:
        # addestramento del modello XGBoost
            param = {
                'n_estimators': 115,
                'max_depth': 28,
                'reg_alpha': 1,
                'reg_lambda': 3,
                'min_child_weight': 5,
                'gamma': 0,
                'eta': 0.31,
                'colsample_bytree': 0.11,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False
            }
            modello = XGBClassifier(**param)
        modello.fit(train_data, train_label)

        # calcolo lo score e lo salvo in un file e in un array per il calcolo della varianza
        prediction = modello.predict(test_data)
        test_score = accuracy_score(test_label, prediction)
        testo = "\n Test " + str(rep) + "; score: " + str(test_score)
        path = "results/xgBoost/scores.txt"
        file = open(path, "a")
        file.write(testo)
        self.scoreList.append(test_score)

        cm = confusion_matrix(test_label, prediction)
        self.confusion_matrix = np.add(self.confusion_matrix, cm)
