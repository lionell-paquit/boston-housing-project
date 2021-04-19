import pandas as pd

from sklearn.model_selection import train_test_split


class DataPipeline:
    def __init__(self, file = '../data/input/boston_corrected.txt'):
        self.data = pd.read_csv(file, header=4, index_col='OBS.', delimiter="\t")
        self.feature_names = ['LSTAT', 'PTRATIO', 'TAX', 'AGE', 'RM', 'NOX', 'INDUS', 'CRIM']
        
    def __features(self):
        return self.data.loc[:,self.feature_names].values
    
    def __target(self):
        return self.data['CMEDV'].values
        
    def clean_data(self):
        # remove unnecessary columns
        columns = ['TOWN', 'TOWN#', 'TRACT', 'LON','LAT', 'MEDV']
        self.data = self.data.drop(labels = columns, axis = 1)
        
        # remove CMEDV outliers
        self.data = self.data[~(self.data.CMEDV >= 50.0)]
        self.data.reset_index(inplace=True, drop=True)
        
    def get_data(self):
        
        self.clean_data()
        X = self.__features()
        y = self.__target()
        
        return train_test_split(X, y, test_size = 0.2, random_state = 1234)