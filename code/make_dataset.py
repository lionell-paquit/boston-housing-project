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
        
    def get_data(self):
        X = self.__features()
        y = self.__target()
        
        return train_test_split(X, y, test_size = 0.2, random_state = 1234)