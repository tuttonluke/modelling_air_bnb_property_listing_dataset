# %%
import datetime
import json
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

def save_model(model, hyperparams: dict, metrics: dict):
    """Saves the model, hyperparamers, and metrics in designated folder.

    Parameters
    ----------
    model : Trained model.
        Model to be saved.
    hyperparams : dict
        Dictionary of best hyperparameters.
    metrics : dict
        Dictionary of performance metrics.
    """ 
    # create a folder with current date and time to save the model in 
    current_time = str(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
    folder_path = f"neural_networks/regression/{current_time}"
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