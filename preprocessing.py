from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class DataPreProcessing:
    def __init__(self, path):
        self.data = read_csv(path)
        self.processedData_path = None

    def preProcessing(self):
        # elimino le colonne MONTH e DAY
        self.data = self.data.drop(['MONTH', 'DAY'], axis=1)

        # elimino le righe dei weekend (weekend = 1) e la relativa colonna
        self.data = self.data.loc[self.data['WEEKEND'] == 0]
        self.data = self.data.drop(['WEEKEND'], axis=1)

        # elimino tutte le righe vuote (tutti i valori a 0
        self.data = self.data.loc[
            (self.data[['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7',
                           'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17',
                           'H18', 'H19', 'H20', 'H21', 'H22', 'H23']] != 0).any(axis=1)]

        # numero i case study
        encoder = LabelEncoder()
        self.data['CASESTUDY'] = encoder.fit_transform(self.data['CASESTUDY'])

    def dataNormalization(self):
        scaler = MinMaxScaler()

        # seleziono solo i valori orari (da H0 ad H23)
        x = self.data.iloc[:, 1:]
        self.data[['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8',
                   'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18',
                   'H19', 'H20', 'H21', 'H22', 'H23']] = scaler.fit_transform(x.T[:]).T

        self.processedData_path = 'data/processedData.csv'
        self.data.to_csv(self.processedData_path, index=False)
