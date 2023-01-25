# %%
from modelling import read_in_data, split_data
from modelling import sklearn_tune_hyperparameters_and_cv, plot_predictions, save_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
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
# %%
if __name__ == "__main__":
    # DECISION TREE TEST
    decision_tree_hyperparams = {
        "max_depth" : [3, 5, 10, None],
        "min_samples_split" : [2, 4, 6, 8]
    }
    best_decision_tree_hyperparams, decision_tree_score = decision_tree_test(decision_tree_hyperparams)

    # RANDOM FOREST TEST
    random_forest_hyperparams = {
        "max_depth" : [3, 5, 10, None],
        "min_samples_split" : [2, 4, 6, 8],
        "max_features" : [2, 3, 4, 5]
    }
    best_random_forest_hyperparams, random_forest_score = random_forest_test(random_forest_hyperparams)

    # GRADIENT BOOST TEST
    gradient_boost_hyperparams = {
        "learning_rate" : [0.05, 0.1],
        "n_estimators" : [50, 100, 200],
        "max_depth" : [3, 5, 8],
        "min_samples_split" : [2, 4],
        "max_features" : [2, 3, 4, 5]
    }
    best_gradient_boost_hyperparams, gradient_boost_score = gradient_boost_test(gradient_boost_hyperparams)


