# library doc string
"""
Author: Yuliia Khomenko
Date Created: 3 Apr 2022

This file contains script to perform EDA, feature engineering, models
training and results reporting
"""
import logging
import joblib
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve
from helpers import *

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plot_and_save(df['Churn'], title='Customer Churn Histogram')
    plot_and_save(
        df['Customer_Age'],
        title='Customer Age Distribution Histogram')
    plot_and_save(df['Marital_Status'].value_counts('normalize'), kind='bar',
                  title='Customer_Marital_Status_Distribution_Barplot')
    plot_and_save(df['Total_Trans_Ct'],
                  title='Total Transport Cost Distribution Histogram')
    plt.figure(figsize=(20, 10))
    HEATMAP_PTH = os.path.join(IMG_PATH, generate_img_name())
    sns.heatmap(df.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2).get_figure().savefig(HEATMAP_PTH)
    plt.close()
    logging.info("SUCCESS: EDA performed successfully,plots are saved")
    return df


def encode_categorical_features(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables orindex y column]

    output:
            df: pandas dataframe with new columns for
    """
    for category in category_lst:
        category_values = []
        category_group_means = df.groupby(category).mean()['Churn']
        for val in df[category]:
            category_values.append(category_group_means.loc[val])
        if response:
            df[response] = category_values
        else:
            response_col_name = category + '_' + 'Churn'
            df[response_col_name] = category_values
    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    df = encode_categorical_features(df, cat_columns, response=None)
    if response:
        y = df[response]
    else:
        y = df['Churn']
    X = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed)
    logging.info("Training and testing datasets are generated")
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    generate_report('Random Forest Train', 'Random Forest Test',
                    y_train, y_train_preds_rf,
                    y_test, y_test_preds_rf)
    generate_report('Logistic Regression Train', 'Logistic Regression Test',
                    y_train, y_train_preds_lr, y_test, y_test_preds_lr)


def feature_importance_plot(model, X_data, image_path):
    """
    creates and stores the feature importances plot in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    names, importances, indices = calculate_feature_importances(model, X_data)
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    IMPORTANCES_PLOT_PTH = os.path.join(image_path, generate_img_name())
    plt.savefig(IMPORTANCES_PLOT_PTH)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)
    fig = plot_roc_curve(lrc, X_test, y_test)
    fig = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=fig.ax_,
        alpha=0.8)
    ROC_CURVE_PLOT_PTH = os.path.join(IMG_PATH, generate_img_name())
    plt.savefig(ROC_CURVE_PLOT_PTH)
    plt.close()
    RF_PATH = os.path.join(MODEL_PATH, 'rfc_model.pkl')
    LR_PATH = os.path.join(MODEL_PATH, 'lr_model.pkl')
    joblib.dump(cv_rfc.best_estimator_, RF_PATH)
    joblib.dump(lrc, LR_PATH)


def generate_predictions(model, X_train, X_test):
    """
    Generates predictions on trainin and testing datasets
    :param model: sklearn model object
    :param X_train: training dataset
    :param X_test: testing dataset
    :return: train_preds : array of predictions on training dataset
            test_preds : array of predictions on testing dataset
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    return train_preds, test_preds


if __name__ == "__main__":
    # read csv into dataframe
    data = import_data(DATA_PATH)
    # basic EDA + save plots
    data = perform_eda(data)
    # feature engineering
    data_train, data_test, labels_train, labels_test = perform_feature_engineering(
        data, response=None)
    # train and save models
    train_models(data_train, data_test, labels_train, labels_test)
    rfc_model = joblib.load(os.path.join(MODEL_PATH, 'rfc_model.pkl'))
    lr_model = joblib.load(os.path.join(MODEL_PATH, 'lr_model.pkl'))
    # calculate feature importances
    feature_importance_plot(rfc_model, data[keep_cols], IMG_PATH)
    train_preds_lr, test_preds_lr = generate_predictions(
        rfc_model, data_train, data_test)
    train_preds_rf, test_preds_rf = generate_predictions(
        lr_model, data_train, data_test)
    # produce and save classification report
    classification_report_image(
        labels_train,
        labels_test,
        train_preds_lr,
        train_preds_rf,
        test_preds_lr,
        test_preds_rf)
