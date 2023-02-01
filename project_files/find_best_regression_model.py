# %%
from regression_modelling import read_in_data, split_data
from regression_modelling import sklearn_tune_hyperparameters_and_cv, plot_predictions, save_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import os
# %%
def regression_model_test(model: str, hyperparam_dict: dict):
    """Fits model to data and tunes hyperparameters, returning the
    model with the highest score.

    Parameters
    ----------
    hyperparam_dict : dict
        Dictionary of hyperparameters to be tuned.

    Returns
    -------
    tuple
        First item a dictionary of optimal hyperparameters,
        second item a scaler of best validation score.
    """
    # read in and split data
    feature_df_scaled, label_series  = read_in_data()
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(feature_df_scaled, label_series)
    
    # select and initialise model, tune hyperparameters
    if model == "decision_tree":
        best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(DecisionTreeRegressor(random_state=1),
                                                                feature_df_scaled,
                                                                label_series,
                                                                hyperparam_dict)
        regressor = DecisionTreeRegressor(**best_hyperparams, random_state=1)
    
    elif model == "random_forest":
        best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(RandomForestRegressor(random_state=1),
                                                                feature_df_scaled,
                                                                label_series,
                                                                hyperparam_dict)
        regressor = RandomForestRegressor(**best_hyperparams, random_state=1)
    
    elif model == "gradient_boost":
        best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(GradientBoostingRegressor(random_state=1),
                                                                feature_df_scaled,
                                                                label_series,
                                                                hyperparam_dict)
        regressor = GradientBoostingRegressor(**best_hyperparams, random_state=1)  

    else:
        print("Invalid model.")
        return

    # fit model and predict validation set labels
    regressor.fit(X_train, y_train)
    y_validation_pred = regressor.predict(X_validation)

    plot_predictions(y_validation, y_validation_pred)
    
    print(f"Best {model} hyperparameters: {best_hyperparams}")
    print(f"Best {model} score: {round(best_score, 3)}")

    # save model
    folder_path = f"models/regression/{model}"
    os.mkdir(folder_path)
    save_model(regressor, best_hyperparams, best_score, folder_path)

    return best_hyperparams, best_score

def find_best_regression_model(model_dict: dict):
    """Finds the model with the best accuracy score.

    Parameters
    ----------
    model_dict : dict
        Dictionary of models tested: keys are model names and values are lists with 
        first item a dictionary of hyperparameters and second item a scalar of the
        accuracy score.

    Returns
    -------
    tuple
        First item is the name of the best model.
        Second item is the accuracy score of the best model.
    """
    best_score = 0
    best_model = None    
    
    # loop through models_and_hyperparams dictionary to find the best model with
    # optimal hyperparameters
    for key, value in model_dict.items():
        if value[1] > best_score:
            best_score = value[1]
            best_model = key
    print(f"\nThe best model is {best_model} with a validation score of {round(best_score, 4)}.")

    return best_model, best_score
# %%
if __name__ == "__main__":

    models_and_hyperparams = {
        "decision_tree" : {
                            "max_depth" : [3, 5, 10, None],
                            "min_samples_split" : [2, 4, 6, 8]
                            },
        "random_forest" : {
                            #"n_estimators" : [100, 200],
                            "max_depth" : [3, 5, None],
                            #"min_samples_split" : [2, 4, 6, 8],
                            #"max_features" : [2, 3, 4]
                            },
        "gradient_boost" : {
                            "learning_rate" : [0.05, 0.1],
                            #"n_estimators" : [100, 200],
                            #"max_depth" : [3, 5],
                            #"min_samples_split" : [2, 4],
                            #"max_features" : [2, 3, 4]
                            }   
    }

    model_results_dict = {}
    # loop through models_and_hyperparams dictionary to find the best model with
    # optimal hyperparameters
    for model, hyperparam_dict in models_and_hyperparams.items():
        best_hyperparams, performance_metrics = regression_model_test(model, hyperparam_dict)
        model_results_dict[model] = [best_hyperparams, performance_metrics]

    best_model, best_score = find_best_regression_model(model_results_dict)