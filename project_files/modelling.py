# %%
from read_tabular_data import TabularData
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing
# %%
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
    X_train, X_test, y_train, y_test = train_test_split(feature_dataframe, 
                                                    label_series, 
                                                    test_size=test_size
                                                    )
    X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                            y_test,
                                                            test_size=test_size
                                                            )                                                    
    return X_train, y_train, X_test, y_test, X_validation, y_validation


def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    """Generator which sequentially yields dictionaries accounting for all 
    hyperparameter combinations in the given grid. 

    Parameters
    ----------
    hyperparameters : typing.Dict[str, typing.Iterable]
        Dictionary of hyperparameters. Key is hyperparameter name, value is 
        an iterable of values to be tested.

    Yields
    ------
    dict
        dictionary of hyperparameter combinations to be tested.
    """
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def custom_tune_regression_model_hyperparameters(model, 
                                                feature_dataframe: pd.DataFrame,
                                                hyperparameter_dict: dict):
    # split data into train, test, and validation sets

    # set initial values for best hyperparameter combination and best loss                                                
    best_hyperparams, best_loss = None, np.inf
    for hyperparams in grid_search(hyperparameter_dict):
        model.fit()
# %%
if __name__ == "__main__":
    # load in data
    np.random.seed(20)
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()

    feature_df, label_series = tabular_df.load_airbnb(
        numerical_tabular_df,
        label="Price_Night"
    )
    feature_df = feature_df.drop("ID", axis=1)

    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(
                                                                        feature_df,
                                                                        label_series,
                                                                        test_size=0.3
    )

    model = make_pipeline(StandardScaler(), SGDRegressor())
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    mse_loss_train = metrics.mean_squared_error(y_train, y_train_pred)
    mse_loss_test = metrics.mean_squared_error(y_test, y_test_pred)
    mse_loss_validation = metrics.mean_squared_error(y_validation, y_validation_pred)
    print("Mean Squared Error Loss on Training data: ", mse_loss_train)
    print("Mean Squared Error Loss on Test data: ", mse_loss_test)
    print("Mean Squared Error Loss on Validation data: ", mse_loss_validation)

    plt.figure()
    plt.scatter(np.arange(50), y_validation_pred[:50], c="r", label="Predictions")
    plt.scatter(np.arange(50), label_series[:50], c="b", label="True Labels", marker="x")
    plt.legend()
    plt.xlabel("Sample Numbers")
    plt.ylabel("Values")
    plt.show()

    print(f"Root mean squared error for training set: {mse_loss_train**0.5}")
    print(f"Root mean squared error for test set: {mse_loss_test**0.5}")

    r_squared_train = metrics.r2_score(y_train, y_train_pred)
    r_squared_test = metrics.r2_score(y_test, y_test_pred)
    print(f"R^2 for training set: {r_squared_train}")
    print(f"R^2 for test set: {r_squared_test}")

# %%