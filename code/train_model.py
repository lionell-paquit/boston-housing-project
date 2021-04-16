# Import required libraries
import os
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform, randint

from xgboost import XGBRegressor
from make_dataset import DataPipeline

class ModelEvaluation:
    def __init__(self):
        pipeline = DataPipeline()
        self.X_train, self.X_test, self.y_train, self.y_test = pipeline.get_data()
        
    def performance_metrics(self, y_true, y_pred):
        rmse = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
        r2 = np.round(r2_score(y_true, y_pred), 3)

        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")
        
    def display_scores(self, scores):
        print("RMSE Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(np.sqrt(scores), np.mean(scores), np.std(scores)))
    
    def report_best_scores(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("---")
    
    def cross_validate(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

        scores = []

        for train_index, test_index in kfold.split(self.X_train):   
            xgb_model = XGBRegressor(n_jobs=1).fit(self.X_train[train_index], self.y_train[train_index])
            
            predictions = xgb_model.predict(self.X_train[test_index])
            actuals = self.y_train[test_index]

            scores.append(mean_squared_error(actuals, predictions))

        self.display_scores(scores)
        
    def evaluate(self, model):
        # model evaluation for training set
        y_train_pred = model.predict(self.X_train)

        print("The model performance for training set")
        self.performance_metrics(self.y_train, y_train_pred)

        # model evaluation for testing set
        y_test_pred = model.predict(self.X_test)

        print("The model performance for testing set")
        self.performance_metrics(self.y_test, y_test_pred)
        
        # Cross-validation
        print("Performing cross-validation on our dataset.")
        self.cross_validate()

class ModelTraining:
    def __init__(self, tuning = True, random_state = None):
        self.tuning = tuning
        self.random_state = random_state
        self.model = XGBRegressor()
        self.params = {
            "n_estimators": randint(100, 150), # default 100
            "max_depth": randint(2, 6), # default 3
            "learning_rate": uniform(0.03, 0.3), # default 0.1
            "gamma": uniform(0, 0.5),
            "subsample": uniform(0.6, 0.4) ,
            "colsample_bytree": uniform(0.7, 0.3)
        }
        
        pipeline = DataPipeline()
        self.X_train, self.X_test, self.y_train, self.y_test = pipeline.get_data()
                
    # Search for the best parameters and return the best estimator
    def tune_parameters(self, model):
        tuned_model = RandomizedSearchCV(model, param_distributions=self.params, random_state=self.random_state, n_iter=200, cv=5, verbose=1, n_jobs=1, return_train_score=True)
        
        return tuned_model
    
    def report_best_scores(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("---")
                
    def save_model(self, model):
        print('INFO: Saving model to ../model/model.pkl')
        pickle.dump( model, open('../model/model.pkl', 'wb') )
        
    def run(self):
        # Do hyperparameter searching if tuning is set
        if self.tuning:
            print("INFO: Tuning model parameters...")
            self.model = self.tune_parameters(self.model)
            self.model.fit(self.X_train, self.y_train)
            self.report_best_scores(self.model.cv_results_, 1)
        else:  
            self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        print("INFO: Model performance evaluation...")
        performance = ModelEvaluation()
        performance.evaluate(self.model)
        
        # save the model
        self.save_model(self.model)

if __name__ == "__main__":
    model_training = ModelTraining(tuning = True, random_state = 1234)
    model_training.run()