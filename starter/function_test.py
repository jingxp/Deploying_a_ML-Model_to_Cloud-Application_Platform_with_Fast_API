# Script to test functions used during modeling.

import pandas as pd
import pytest
from starter.ml.data import process_data, data_slice
from starter.ml.model import compute_model_metrics

@pytest.fixture
def cat_features():
    categorical_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                    ]
    return categorical_features

@pytest.fixture
def test_df():
    """Using the original data set for testing"""
    sample_df = pd.read_csv('../data/census_clean.csv')
    return sample_df

def test_process_data(test_df, cat_features):
    """Test the data process fuction"""
    test_data = test_df
    cat_features = cat_features
    X_train, y_train, encoder, lb = process_data(
        test_data, categorical_features=cat_features, label="salary", training=True
    )
    assert len(X_train) > 0

def test_data_slice(test_df):
    """Test the data slice function"""
    test_data = test_df
    slice_feature = "sex"
    sliced_data = data_slice(test_data,slice_feature)
    assert len(sliced_data) > 0
    assert all(len(item) > 0 for item in sliced_data.values())

def test_compute_model_metrics():
    """Test the computation of metrics using dummy data"""
    labels = [0,0,0,1,1,1,0,0,0,1]
    preds =  [1,0,1,0,1,1,1,0,0,0]
    precision, recall, fbeta = compute_model_metrics(labels, preds)
    assert precision == 0.4
    assert recall == 0.5