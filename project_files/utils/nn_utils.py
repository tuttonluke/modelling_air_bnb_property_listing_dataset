# %%
import datetime
import json
import numpy as np
import os
import pickle
import random
import yaml
# %%
def generate_nn_configs(n_configs: int) -> list:
    """Generates a list of configuration dictionaries.

    Parameters
    ----------
    n_configs : int
        Number of configurations to be created.

    Returns
    -------
    list
        List of configuration dictionaries.
    """
    dict_list = []
    # generate values for applicable hyperparameters
    for i in range(n_configs):
        learning_rate = random.choice([1/i for i in [10**j for j in range(3, 4)]])
        epochs = random.choice([i for i in range(5, 30)])
        config_dict = {
            "learning_rate" : learning_rate,
            "epochs" : epochs
        }
        dict_list.append(config_dict)

    return dict_list
    
def get_nn_config(file_path: str) -> dict:
    """Reads a YAML file containing neural netweork 
    configuration parameters and returns them in a dictionary.

    Parameters
    ----------
    file_path : str
        File path of YAML file.

    Returns
    -------
    dict
        Dictionary of configuration parameters.
    """
    with open(f"network_configs/{file_path}", "r") as file:
        try:
            config_dict = yaml.safe_load(file)
        except yaml.YAMLError as error:
            print(error)
    return config_dict


def save_configs_as_yaml(config_list: list):
    """Takes a list of configuration dictionaries and saves each item
    individually as a YAML file.

    Parameters
    ----------
    config_list : list
        List of configuration dictionaries.
    """
    for idx, config in enumerate(config_list):
        with open(f"network_configs/{idx}.yaml", "w") as file:
            yaml.dump(config, file)

def save_model(model, hyperparams: dict, metrics: dict, model_type: str):
    """Saves the model, hyperparamers, and metrics in designated folder.

    Parameters
    ----------
    model : Trained model.
        Model to be saved.
    hyperparams : dict
        Dictionary of best hyperparameters.
    metrics : dict
        Dictionary of performance metrics.
    model_type: str
        "regression" or "classificaiton".
    """ 
    # create a folder with current date and time to save the model in 
    current_time = str(datetime.datetime.now()).replace(" ", "_").replace(":", ".")

    if model_type == "regression":
        folder_path = f"deep_learning_models/regression/{current_time}"
    elif model_type == "classification":
        folder_path = f"deep_learning_models/classification/{current_time}"
    else:
        print("Invalid model type. Enter 'classification' or 'regression'.")
        return
    os.mkdir(folder_path)   
    
    # save model
    with open(f"{folder_path}/model.pt", "wb") as file:
        pickle.dump(model, file)
    
    # save hyperparameters
    with open(f"{folder_path}/hyperparams.json", "w") as file:
        json.dump(hyperparams, file)

    # save model metrics in json file
    with open(f"{folder_path}/metrics.json", "w") as file:
        json.dump(metrics, file)

def find_best_regression_nn():
    """Cycles through all metrics json files in designated config_directory
    and prints the metrics and file location of the model with the highest
    mean squared error and r_squares scores.
    """
    best_mse = np.inf
    best_r2 = -np.inf
    best_mse_model = None
    best_r2_model = None

    config_directory = r"deep_learning_models\regression"
    for idx, (root, dirs, files) in enumerate(os.walk(config_directory)):
        for file in files:
            if file == "metrics.json":
                with open(f"{root}\{file}") as metrics_json:
                    metrics_dict = json.load(metrics_json)
                    # update best model for MSE score
                    if metrics_dict["test_MSE"] < best_mse:
                        best_mse = metrics_dict["test_MSE"]
                        best_mse_model = f"{idx-1}, {root[32:]}"
                    # update best model for r_squared score
                    if metrics_dict["test_r_squared"] > best_r2:
                        best_r2 = metrics_dict["test_r_squared"]
                        best_r2_model = f"{idx-1}, {root[32:]}"
    
    # Print scores and location of best model                    
    print(f"Best MSE is {best_mse:.2f}, model {best_mse_model}")
    print(f"Best r_squared score is {best_r2:.4f}, model {best_r2_model}")

def find_best_classification_nn():
    """Cycles through all metrics json files in designated config_directory
    and prints the metrics and file location of the model with the highest
    mean squared error and r_squares scores.
    """
    best_f1 = -np.inf
    best_f1_model = None

    config_directory = r"deep_learning_models\classification"
    for idx, (root, dirs, files) in enumerate(os.walk(config_directory)):
        for file in files:
            if file == "metrics.json":
                with open(f"{root}\{file}") as metrics_json:
                    metrics_dict = json.load(metrics_json)
                    # update best model for MSE score
                    if metrics_dict["f1_macro"] < best_f1:
                        best_f1 = metrics_dict["f1_macro"]
                        best_f1_model = f"{idx-1}, {root[32:]}"
    
    # Print scores and location of best model                    
    print(f"Best F1 score is {best_f1:.2f}, model {best_f1_model}")