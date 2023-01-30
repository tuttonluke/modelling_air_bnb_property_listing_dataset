# %%
from classification_modelling import read_in_data, sklearn_tune_hyperparameters_and_cv, visualise_confusion_matrix
from find_best_regression_model import find_best_model
from regression_modelling import split_data, plot_predictions, save_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import os
# %%
def model_test(model: str, hyperparam_dict: dict):
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
        second item a dictionary of performance metrics.
    """
    # read in and split data
    feature_df_scaled, label_series  = read_in_data()
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(feature_df_scaled, label_series)
    
    if model == "decision_tree":
        best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(DecisionTreeClassifier(random_state=1),
                                                                feature_df_scaled,
                                                                label_series,
                                                                hyperparam_dict)
        classifier = DecisionTreeClassifier(**best_hyperparams, random_state=1)
    
    elif model == "random_forest":
        best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(RandomForestClassifier(random_state=1),
                                                                feature_df_scaled,
                                                                label_series,
                                                                hyperparam_dict)
        classifier = RandomForestClassifier(**best_hyperparams, random_state=1)
    
    elif model == "gradient_boost":
        best_hyperparams, best_score = sklearn_tune_hyperparameters_and_cv(GradientBoostingClassifier(random_state=1),
                                                                feature_df_scaled,
                                                                label_series,
                                                                hyperparam_dict)
        classifier = GradientBoostingClassifier(**best_hyperparams, random_state=1)  

    else:
        print("Invalid model.")
        return
    
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    y_validation_pred = classifier.predict(X_validation)

    plot_predictions(y_validation, y_validation_pred)

    # evaluate statistics
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

    # # save model
    # folder_path = f"models/classification/{model}"
    # os.mkdir(folder_path)
    # save_model(classifier, best_hyperparams, performance_metrics, folder_path)

    return best_hyperparams, performance_metrics

def find_best_classification_model(model_dict: dict):
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
        if value[1]["Validation F1 score"] > best_score:
            best_score = value[1]["Validation F1 score"]
            best_model = key
    print(f"The best model is {best_model} with an F1 score of {round(best_score, 4)}.")

    return best_model, best_score
# %%
if __name__ == "__main__":
    
    models_and_hyperparams = {
        "decision_tree" : {
                            "max_depth" : [3, 5, 10, None],
                            "min_samples_split" : [2, 4, 6, 8]
                            },
        "random_forest" : {
                            "n_estimators" : [100, 200, 300],
                            "max_depth" : [3, 5, 10, None],
                            "min_samples_split" : [2, 4, 6, 8],
                            "max_features" : [2, 3, 4, 5]
                            },
        "gradient_boost" : {
                            "learning_rate" : [0.05, 0.1],
                            "n_estimators" : [50, 100, 200],
                            "max_depth" : [3, 5, 8],
                            "min_samples_split" : [2, 4],
                            "max_features" : [2, 3, 4, 5]
                            }   
    }
    
    model_results_dict = {}

    for model, hyperparameters in models_and_hyperparams.items():
        best_hyperparams, performance_metrics = model_test(model, hyperparameters)
        model_results_dict[model] = [best_hyperparams, performance_metrics]

    best_model, best_score = find_best_classification_model(model_results_dict)

# TODO file not found?
# TODO consolidate regression test functions
