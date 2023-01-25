# %%
from modelling import read_in_data, split_data
from modelling import sklearn_tune_hyperparameters_and_cv, plot_predictions, save_model
from sklearn.tree import DecisionTreeRegressor
import os
# %%
def decision_tree_test(hyperparam_dict):
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
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best score: {round(best_score, 3)}")

    # save model
    os.mkdir("models/regression/decision_trees")
    save_model(regressor, best_hyperparams, best_score, "models/regression/decision_trees")

    return best_hyperparams, best_score
# %%
if __name__ == "__main__":
    # DECISION TREE TEST
    decision_tree_hyperparams = {
        "max_depth" : [3, 5, 10, None],
        "min_samples_split" : [2, 4, 6, 8]
    }
    decision_tree_hyperparams, decision_tree_score = decision_tree_test(decision_tree_hyperparams)