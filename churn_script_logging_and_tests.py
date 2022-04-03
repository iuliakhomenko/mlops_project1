"""
Author: Yuliia Khomenko
Date Created: 3 Apr 2022

This file contains tests of the main functions of churn_library.py
"""
import os
import logging
import pytest
import pandas as pd
import joblib
from churn_library import import_data, perform_eda, encode_categorical_features,\
    perform_feature_engineering
from constants import DATA_PATH, IMG_PATH, LOG_PATH, cat_columns, keep_cols, MODEL_PATH


@pytest.fixture(scope="module")
def path():
    """
    pytest fixture that allows to share path to the dataset between
    testing functions
    input: None

    output: path to the data
    """
    return DATA_PATH


@pytest.fixture(scope="module")
def data():
    """
    pytest fixture that allows to share original dataset between
    testing functions
    input: None

    output: original dataset in pandas dataframe format
    """
    df = pd.read_csv(DATA_PATH)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import(path):
    """
    test data import - this example is completed for you to assist with the other test functions
    input: path to the datatset csv file
    output: None
    """
    try:
        df = import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing test_import: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("imported dataframe is not empty")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    try:
        assert df.isnull().sum().sum() == 0
        logging.info("imported datafarme has no null values")
    except AssertionError as err:
        logging.error("There are null values in your dataframe!")
        raise err


def test_eda(data):
    '''
    test perform eda function
    input: dataset in pandas dataframe format
    output: None
    '''
    perform_eda(data)
    try:
        assert len(os.listdir(IMG_PATH)) > 0
        assert len(os.listdir(IMG_PATH)) == 5
        logging.info('All EDA plots were saved to image directory')
    except AssertionError as err:
        logging.error('Not all plots were saved to image directory')
        raise err


def test_encoder_categorical_features(data):
    '''
    test encoder helper
    input: dataset in pandas dataframe format
    output: None
    '''
    df = data
    cols_list = list(df.columns)
    df = encode_categorical_features(df, cat_columns, response=None)
    new_cols_list = list(df.columns)
    columns_generated = list(set(new_cols_list) - set(cols_list))
    # check if columns were generated
    try:
        assert len(columns_generated) == len(cat_columns)
    except AssertionError as err:
        logging.error(
            'Errors in categorical features encoder. Columns not generated properly')
        raise err
    # check if there are no nulls in data
    try:
        assert df.isnull().sum().sum() == 0
    except AssertionError as err:
        logging.error('df contains null values')
        raise err


def test_perform_feature_engineering(data):
    """
	test perform_feature_engineering
	input: dataset in pandas dataframe format
	output: None
	"""
    # check the shape of the dataframe
    df = data
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, None)
    try:
        assert x_train.shape[1] == df[keep_cols].shape[1]
    except AssertionError as err:
        logging.error("Not all categorical columns were encoded")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        joblib.load(os.path.join(MODEL_PATH, 'rfc_model.pkl'))
        joblib.load(os.path.join(MODEL_PATH, 'lr_model.pkl'))
        logging.info('Testing train_models: SUCCESS')
    except FileNotFoundError as err:
        logging.error('Testing train_models: the file is not found')
        raise err
