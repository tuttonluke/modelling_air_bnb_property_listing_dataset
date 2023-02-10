# %%
from read_tabular_data import TabularData
from regression_modelling import normalise_data, split_data
from regression_modelling import plot_predictions, save_model
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
# %%
def read_in_data():
    """Reads in, cleans, splits, and normalises data for analysis.

    Returns
    -------
    tupple
        Tuple containing DataFrames of features and labels.
    """
    tabular_df = TabularData()
    classification_tabular_df = tabular_df.get_classification_data()
    feature_df, label_series = tabular_df.load_airbnb(
            classification_tabular_df,
            label="Category"
        )
    feature_df_scaled = normalise_data(feature_df)

    return feature_df_scaled, label_series

def visualise_confusion_matrix(y_true, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()

def sklearn_tune_hyperparameters_and_cv(model, x, y, hyperparam_grid):
    """Tunes hyperparameters using grid search and k-fold cross validation, scored with f1 metric.

    Parameters
    ----------
    model : class
        Classification model
    x : pd.DataFrame
        DataFrame of model features.
    y : pd.Series
        Series of model labels.
    hyperparameter_dict : dict
        Dictionary of hyperparameter combinations to be tested.

    Returns
    -------
    tuple
        Dictionary of best parameters and scalar value of best train f1 score.
    """
    np.random.seed(42)
    # perform cross validation and hyperparameter tuning
    model_cv = GridSearchCV(model, hyperparam_grid, cv=5, scoring="f1_macro")
    model_cv.fit(x, y)
    best_params = model_cv.best_params_
    best_score = model_cv.best_score_

    return best_params, best_score
# %%
if __name__ == "__main__":
    # Surpress Convergence warning for this model - logistic regression will not converge
    # on this data!
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    # load in and normalise data
    feature_df_scaled, label_series = read_in_data()
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(feature_df_scaled, label_series)

    # hyperparameter tuning and cross validation
    hyperparam_grid = {
        "max_iter" : [100, 200, 500, 1000, 2000],
    }
    best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(LogisticRegression(),
                                                        feature_df_scaled,
                                                        label_series,
                                                        hyperparam_grid)
    # initialise and fit model
    model = LogisticRegression(**best_hyperparams)
    model.fit(X_train, y_train)

    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    plot_predictions(y_validation, y_validation_pred)

    visualise_confusion_matrix(y_test, y_test_pred)

    print(best_hyperparams)
    performance_metrics = {
        "Test Accuracy" : round(metrics.accuracy_score(y_test, y_test_pred), 3),
        "Test Precision" : round(metrics.precision_score(y_test, y_test_pred, average="macro"), 3),
        "Test Recall" : round(metrics.recall_score(y_test, y_test_pred, average="macro"), 3),
        "Test F1 score" : round(metrics.f1_score(y_test, y_test_pred, average="macro"), 3),
        "Validation Accuracy" : round(metrics.accuracy_score(y_validation, y_validation_pred), 3),
        "Validation Recall" : round(metrics.recall_score(y_validation, y_validation_pred, average="macro"), 3),
        "Validation F1 score" : round(metrics.f1_score(y_validation, y_validation_pred, average="macro"), 3)
    }
    print(f"\n{performance_metrics}")
    # os.mkdir("models/classification/logistic_regression")
    # save_model(model, best_hyperparams, performance_metrics, "models/classification/logistic_regression")
