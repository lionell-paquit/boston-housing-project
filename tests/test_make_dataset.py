import os
import sys

import pytest
import pandas as pd

sys.path.insert(1, '../code/')
from make_dataset import DataPipeline

def test_df_dimensions():
    data_pipeline = DataPipeline()
    expected_before_clean = (506, 20)
    expected_after_clean = (490, 14)
    
    assert data_pipeline.data.shape == expected_before_clean
    data_pipeline.clean_data()
    assert data_pipeline.data.shape == expected_after_clean
    
def test_get_data():
    data_pipeline = DataPipeline()
    expected_train_shape = (392, 8)
    expected_test_shape = (98, 8)
    
    train, test, _, _ = data_pipeline.get_data()
    assert train.shape == expected_train_shape
    assert test.shape == expected_test_shape
    