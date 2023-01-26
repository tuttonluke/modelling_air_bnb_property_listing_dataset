# %%
from regression_modelling import read_in_data, split_data
from regression_modelling import sklearn_tune_hyperparameters_and_cv, plot_predictions, save_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import os
# %%
def decision_tree_test(hyperparam_dict: dict):
    """Fits decision tree model to data and tunes hyperparameters, returning the
    model with the highest score.

    Parameters
    ----------
    hyperparam_dict : dict
        Dictionary of hyperparameters to be tuned.

    Returns
    -------
    tuple
        First item a dictionary of optimal hyperparameters,
        second item a scaler of the best score associated with the model with these hyperparameters.
    """
    # read in and split data
    feature_df_scaled, label_series  = read_in_data()
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(feature_df_scaled, label_series)
    
    
    best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(DecisionTreeRegressor(random_state=1),
                                                            feature_df_scaled,
                                                            label_series,
                                                            hyperparam_dict)
    regressor = DecisionTreeRegressor(**best_hyperparams, random_state=1)
    regressor.fit(X_train, y_train)
    y_validation_pred = regressor.predict(X_validation)

    plot_predictions(label_series, y_validation_pred)

    # evaluate statistics
    print(f"Best decision tree hyperparameters: {best_hyperparams}")
    print(f"Best decision tree score: {round(best_score, 3)}")

    # save model
    os.mkdir("models/regression/decision_trees")
    save_model(regressor, best_hyperparams, best_score, "models/regression/decision_trees")

    return best_hyperparams, best_score

def random_forest_test(hyperparam_dict: dict):
    """Fits random forest model to data and tunes hyperparameters, returning the
    model with the highest score.

    Parameters
    ----------
    hyperparam_dict : dict
        Dictionary of hyperparameters to be tuned.

    Returns
    -------
    tuple
        First item a dictionary of optimal hyperparameters,
        second item a scaler of the best score associated with the model with these hyperparameters.
    """
    # read in and split data
    feature_df_scaled, label_series  = read_in_data()
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(feature_df_scaled, label_series)
    
    
    best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(RandomForestRegressor(random_state=1),
                                                            feature_df_scaled,
                                                            label_series,
                                                            hyperparam_dict)
    regressor = RandomForestRegressor(**best_hyperparams, random_state=1)
    regressor.fit(X_train, y_train)
    y_validation_pred = regressor.predict(X_validation)

    plot_predictions(label_series, y_validation_pred)

    # evaluate statistics
    print(f"Best random forest hyperparameters: {best_hyperparams}")
    print(f"Best random forest score: {round(best_score, 3)}")

    # save model
    os.mkdir("models/regression/random_forests")
    save_model(regressor, best_hyperparams, best_score, "models/regression/random_forests")

    return best_hyperparams, best_score

def gradient_boost_test(hyperparam_dict):
    """Fits gradient boost model to data and tunes hyperparameters, returning the
    model with the highest score.

    Parameters
    ----------
    hyperparam_dict : dict
        Dictionary of hyperparameters to be tuned.

    Returns
    -------
    tuple
        First item a dictionary of optimal hyperparameters,
        second item a scaler of the best score associated with the model with these hyperparameters.
    """
    # read in and split data
    feature_df_scaled, label_series  = read_in_data()
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(feature_df_scaled, label_series)
    
    
    best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(GradientBoostingRegressor(random_state=1),
                                                            feature_df_scaled,
                                                            label_series,
                                                            hyperparam_dict)
    regressor = GradientBoostingRegressor(**best_hyperparams, random_state=1)
    regressor.fit(X_train, y_train)
    y_validation_pred = regressor.predict(X_validation)

    plot_predictions(label_series, y_validation_pred)

    # evaluate statistics
    print(f"Best gradient boost hyperparameters: {best_hyperparams}")
    print(f"Best gradient boost score: {round(best_score, 3)}")

    # save model
    os.mkdir("models/regression/gradient_boost")
    save_model(regressor, best_hyperparams, best_score, "models/regression/gradient_boost")

    return best_hyperparams, best_score

def find_best_model(model_dict: dict):
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
    for key, value in model_dict.items():
        if value[1] > best_score:
            best_score = value[1]
            best_model = key
    print(f"The best model is {best_model} with a score of {round(best_score, 4)}.")

    return best_model, best_score
# %%
if __name__ == "__main__":
    model_dict = {}
    # DECISION TREE TEST
    decision_tree_hyperparams = {
        "max_depth" : [3, 5, 10, None],
        "min_samples_split" : [2, 4, 6, 8]
    }
    best_decision_tree_hyperparams, decision_tree_score = decision_tree_test(decision_tree_hyperparams)
    model_dict["decision_tree"] = [best_decision_tree_hyperparams, decision_tree_score]

    # RANDOM FOREST TEST
    random_forest_hyperparams = {
        "max_depth" : [3, 5, 10, None],
        "min_samples_split" : [2, 4, 6, 8],
        "max_features" : [2, 3, 4, 5]
    }
    best_random_forest_hyperparams, random_forest_score = random_forest_test(random_forest_hyperparams)
    model_dict["random_forest"] = [best_random_forest_hyperparams, random_forest_score]

    # GRADIENT BOOST TEST
    gradient_boost_hyperparams = {
        "learning_rate" : [0.05, 0.1],
        "n_estimators" : [50, 100, 200],
        "max_depth" : [3, 5, 8],
        "min_samples_split" : [2, 4],
        "max_features" : [2, 3, 4, 5]
    }
    best_gradient_boost_hyperparams, gradient_boost_score = gradient_boost_test(gradient_boost_hyperparams)
    model_dict["gradient_boost"] = [best_gradient_boost_hyperparams, gradient_boost_score]

    best_model, best_score = find_best_model(model_dict)


# %%
# TODO fix the model save for gradient boost