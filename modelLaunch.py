from xgBoost import XGBoostClassifier
from gradientboosting import GBClassifier
from utility import printResults
from preprocessing import DataPreProcessing


class ModelLaunch:
    def __init__(self, data_path):
        preProcessing = DataPreProcessing(data_path)
        preProcessing.preProcessing()
        preProcessing.dataNormalization()
        self.path = preProcessing.processedData_path

    def launch(self, model):
        if model == 'Gradient Boosting':
            classifier = GBClassifier(self.path)
        elif model == 'XGBoost':
            classifier = XGBoostClassifier(self.path)
        else:
            print("Wrong model name\n")
            return
        classifier.training_testing(500, 'tuned')
        printResults(classifier.scoreList, classifier.confusion_matrix, model, 'tuned')
