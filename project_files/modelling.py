# %%
from read_tabular_data import TabularData
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import itertools
import joblib
import json
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
    np.random.seed(20)
    X_train, X_test, y_train, y_test = train_test_split(feature_dataframe, 
                                                    label_series, 
                                                    test_size=test_size
                                                    )
    X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                            y_test,
                                                            test_size=test_size
                                                            )                                                    
    return X_train, y_train, X_test, y_test, X_validation, y_validation

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
        Dictionary of hyperparameter combinations to be tested.
    """
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def custom_tune_regression_hyperparameters(model, 
                                    feature_dataframe: pd.DataFrame,
                                    label_series: pd.Series,
                                    hyperparameter_dict: dict):
    """Tunes hyperparameters given in hyperparameter_dict on given model and data.

    Parameters
    ----------
    model : class
        Regression model
    feature_dataframe : pd.DataFrame
        DataFrame of model features.
    label_series : pd.Series
        Series of model labels.
    hyperparameter_dict : dict
        Dictionary of hyperparameter combinations to be tested.

    Returns
    -------
    tuple
        Tuple comtaining dictionary of best hyperparameter combination with accompanying statistics.
    """
    # split data into train, test, and validation sets
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(
                                                                    feature_dataframe,
                                                                    label_series,
                                                                    test_size=0.3
    )

    # set initial values for best hyperparameter combination and best loss                                                
    best_hyperparams, best_loss = None, np.inf
    performance_metrics = {"train_rmse" : None,
                            "test_rmse" : None,
                            "validation_rmse" : None,
                            "train R^2 score" : None,
                            "test R^2 score" : None,
                            "validation R^2 score" : None}
    # loop over hyperparameter combinations to ascertain best combination
    for hyperparams in grid_search(hyperparameter_dict):
        # scale data and initiate model with specific hyperparameter combination
        model = make_pipeline(StandardScaler(), SGDRegressor(**hyperparams))
        model.fit(X_train, y_train) # fit model to training data
        # evaluate root mean squared error loss on validation set
        y_validation_pred = model.predict(X_validation)
        validation_mse = metrics.mean_squared_error(y_validation, y_validation_pred)
        validation_rmse = validation_mse**0.5
        # update best hyperparameters and best loos based on lowest validation loss
        if validation_rmse < best_loss:
            best_loss = validation_rmse
            best_hyperparams = hyperparams
            # fill preformance_metrics disctionary with metrics from best hyperparams
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            performance_metrics["train_rmse"] = round(metrics.mean_squared_error(y_train, y_train_pred)**0.5, 2)
            performance_metrics["test_rmse"] = round(metrics.mean_squared_error(y_test, y_test_pred)**0.5, 2)
            performance_metrics["validation_rmse"] = round(validation_rmse, 2)
            performance_metrics["train R^2 score"] = round(metrics.r2_score(y_train, y_train_pred), 2)
            performance_metrics["test R^2 score"] = round(metrics.r2_score(y_test, y_test_pred), 2)
            performance_metrics["validation R^2 score"] = round(metrics.r2_score(y_validation, y_validation_pred), 2)

    return best_hyperparams, performance_metrics

def sklearn(model, x, y, hyperparam_grid):
    """Tunes hyperparameters using grid search and k-fold cross validation.

    Parameters
    ----------
    model : class
        Regression model
    x : pd.DataFrame
        DataFrame of model features.
    y : pd.Series
        Series of model labels.
    hyperparameter_dict : dict
        Dictionary of hyperparameter combinations to be tested.

    Returns
    -------
    tuple
        Dictionary of best parameters and scalar value of best train r^2 score.
    """
    np.random.seed(42)
    model_cv = GridSearchCV(model, hyperparam_grid, cv=5)
    model_cv.fit(x, y)
    best_params = model_cv.best_params_
    best_score = model_cv.best_score_

    return best_params, best_score

def plot_predictions(y_true, y_predicted, n_points=50):
    plt.figure()
    plt.scatter(np.arange(n_points), y_true[:n_points], c="b", label="True Labels", marker="x")
    plt.scatter(np.arange(n_points), y_predicted[:n_points], c="r", label="Predictions")
    plt.legend()
    plt.xlabel("Sample Numbers")
    plt.ylabel("Values")
    plt.show()

def save_model(model, hyperparams: dict, metrics: dict, folder: str):
    """Saves the model, hyperparamers, and metrics in designated folder.

    Parameters
    ----------
    model : Trained model.
        Model to be saved.
    hyperparams : dict
        Dictionary of best hyperparameters.
    metrics : dict
        Dictionary of performance metrics.
    folder : str
        Filepath of save location.
    """
    model_name = str(model)
    # save model
    joblib.dump(model, f"{folder}/{model_name}.joblib")
    # save hyperparameters in json file
    with open(f"{folder}/{model_name}_hyperparams.json", "w") as file:
        json.dump(hyperparams, file)
    # save model metrics in json file
    with open(f"{folder}/{model_name}_metrics.json", "w") as file:
        json.dump(metrics, file)
# %%
if __name__ == "__main__":
    # load in and normalise data
    np.random.seed(42)
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()

    feature_df, label_series = tabular_df.load_airbnb(
        numerical_tabular_df,
        label="Price_Night"
    )
    feature_df = feature_df.drop("ID", axis=1)
    feature_df_scaled = normalise_data(feature_df)

    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(
                                                                        feature_df_scaled,
                                                                        label_series,
                                                                        test_size=0.3
    )

    # tune model hyperparameters
    hyperparam_grid = {
        "alpha" : [0.00001, 0.0001, 0.001, 0.01],
        "max_iter" : [1000, 1500, 2000, 5000, 10000],
        "eta0" : [0.001, 0.01, 0.1]
    }

    best_hyperparams, performance_metrics = custom_tune_regression_hyperparameters(SGDRegressor(),
                                                                feature_df_scaled,
                                                                label_series,
                                                                hyperparam_grid)
    print(best_hyperparams)

    # initialise and fit model to training data
    model = SGDRegressor(**best_hyperparams)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_validation_pred = model.predict(X_validation)

    plot_predictions(label_series, y_validation_pred)

    # evaluate statistics
    print(performance_metrics)
    save_model(model, best_hyperparams, performance_metrics, "models/regression")
# %%