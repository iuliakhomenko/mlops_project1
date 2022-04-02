"""
This file contains small helper functions of the main script churn_library.py
"""
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report
import numpy as np
from constants import *


def plot_and_save(pd_series, figure_size=(20, 10), kind='hist', title=None):
    """
    Helper function to plot pandas series and save to a predefined folder
    input:
            pd_series: pandas series to plot
            kind: kind of plot to produce
            title: title to use for the plot
    output:
            None
    """
    PLOT_PTH = os.path.join(IMG_PATH, generate_img_name())
    pd_series.plot(kind=kind,
                   figsize=figure_size,
                   title=title).get_figure().savefig(PLOT_PTH)
    plt.close()


def generate_img_name():
    """
    small helper function using current time timestamp to generate file name of a png file
    :return: str  filename
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    return current_time + '.png'


def generate_report(title_train_report, title_test_report,
                    y_train, y_train_preds, y_test, y_test_preds,):
    """
    generates training and testing report in predefined format
    input:
            title_train_report: how train report should be titled
            title_test_report: how test report should be titled
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
    output:
            None
    """
    plt.rc('figure', figsize=(10, 5))
    plt.text(0.01, 1.25, str(title_train_report), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(title_test_report), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis("off")
    PLOT_PTH = os.path.join(IMG_PATH, generate_img_name())
    plt.savefig(PLOT_PTH)
    plt.close()


def calculate_feature_importances(model, X_data):
    """
    calculate feature importances, arranges them in descend in order
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
    output:
            feature_names, feature_importances, indices
    """
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    return names, importances, indices