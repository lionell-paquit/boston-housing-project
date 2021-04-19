import os
import sys
import pytest

import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

sys.path.insert(1, '../code/')

from train_model import ModelEvaluation, ModelTraining

@pytest.fixture(scope="module")
def model_evaluation():
    return ModelEvaluation()

@pytest.fixture(scope="module")
def model_training_no_tuning():
    return ModelTraining(tuning = False, random_state = 1234)

@pytest.fixture(scope="module")
def model_training_with_tuning():
    return ModelTraining(tuning = True, random_state = 1234)

def test_model_no_tuning(model_training_no_tuning):
    model_training_no_tuning.run()
    assert isinstance(model_training_no_tuning.model, XGBRegressor)

def test_model_with_tuning(model_training_with_tuning):
    model_training_with_tuning.run()
    assert isinstance(model_training_with_tuning.model, RandomizedSearchCV)
    
def test_save_model(model_training_no_tuning):
    model_filepath = '../tests/model/test_model.pkl'
    expected_model = []
    model_training_no_tuning.save_model(expected_model, model_filepath)
    
    load_model = pickle.load(open(model_filepath, 'rb'))
    assert load_model == expected_model