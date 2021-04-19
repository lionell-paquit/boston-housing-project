import os
import pickle

from train_model import ModelTraining

class Predict:
    def __init__(self, model_filepath = '../model/model.pkl'):
        
        if not os.path.exists(model_filepath):
            model_training = ModelTraining(tuning = True, random_state = 1234)
            model_training.run()
        
        self.model = pickle.load(open(model_filepath, 'rb'))
            
    def get_prediction(self, X):
        
        return self.model.predict(X)