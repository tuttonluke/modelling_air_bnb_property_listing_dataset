# %%
from read_tabular_data import TabularData
from regression_modelling import save_model
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
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

def normalise_data(data: pd.DataFrame) -> np.array:
    """
    Normalises input data (DataFrame) with Min-Max scaling.

    Parameters
    ----------
    data : pd.Data
        Features data.

    Returns
    -------
    np.array
        Scaled features data in numpy array format.
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.to_numpy())

    return  data_scaled

def split_data(feature_dataframe, label_series, test_size=0.3):
    """Splits feature dataframe into train, test, and validation sets
    in a proportion of test_size.

    Parameters
    ----------
    feature_dataframe : pd.DataFrame
        DataFrame of model features.
    label_series : pd.Series
        Series of model labels.
    test_size : float, optional
        proportion of data to split into test set and validation set, by default 0.3

    Returns
    -------
    tuple
        tuple of arrays of train, test, and validation data.
    """
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(feature_dataframe, 
                                                    label_series, 
                                                    test_size=test_size
                                                    )
    X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                            y_test,
                                                            test_size=test_size
                                                            )                                                    
    return X_train, y_train, X_test, y_test, X_validation, y_validation

def plot_predictions(y_true, y_predicted, n_points=50):
    plt.figure()
    plt.scatter(np.arange(n_points), y_true[:n_points], c="b", label="True Labels", marker="x")
    plt.scatter(np.arange(n_points), y_predicted[:n_points], c="r", label="Predictions")
    plt.legend()
    plt.xlabel("Sample Numbers")
    plt.ylabel("Values")
    plt.show()

def visualise_confusion_matrix(y_true, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()

def plot_roc_curve(y_true, y_score):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_true, 
                                                                        y_score)
    area_under_curve = metrics.auc(false_positive_rate, true_positive_rate)
    display = metrics.RocCurveDisplay(fpr=false_positive_rate, 
                                        tpr=true_positive_rate, 
                                        roc_auc=area_under_curve,
                                        estimator_name='example estimator')
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
    best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(LogisticRegression(random_state=1),
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
