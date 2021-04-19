import os
import sys
import pytest
import pickle
import numpy as np

sys.path.insert(1, '../code/')
from predict_model import Predict
from sklearn.model_selection import RandomizedSearchCV

@pytest.fixture(scope="module")
def predict():
    return Predict(model_filepath = '../tests/model/model.pkl')

def test_predict_model(predict):
    assert isinstance(predict.model, RandomizedSearchCV)
    
def test_get_prediction(predict):
    data = np.array([ [10.24, 16.1, 337., 31.1, 5.787, 0.398, 3.37, 0.0438],
            [13.35, 20.1, 711., 83.5, 5.983, 0.609, 27.74, 0.1113],
            [9.29, 20.2, 224., 58.5, 5.968, 0.515, 5.19, 0.0615],
            [19.31, 20.2, 666., 98.3, 6.417, 0.713, 18.1, 7.526] ])
    
    expected = np.array([18.7487, 19.4495, 20.9987, 14.3416])
    
    results = predict.get_prediction(data)
    
    assert np.allclose(results, expected)