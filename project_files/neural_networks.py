# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
# %%
class AirBnBNightlyPriceImageDataset(Dataset):
    """Creates a PyTorch dataset  of the AirBnb data that returns a tuple 
    of (features, labels) when indexed.
    """
    def __init__(self, features, labels) -> None:
        super().__init__()
        # assert feature and label sets are the same length
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels
    
    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)

    def __len__(self):
        return len(self.features)

class Network(nn.Module):
    def __init__(self, in_features: int, out_features: int, config_file: str) -> None:
        """Initialiser for neural network class.

        Parameters
        ----------
        in_features : int
            Number of features.
        out_features : int
            Number of labels.
        config_file : str
            YAML file containing network parameters.
        """
        super().__init__()
        self.config_dict = get_nn_config(config_file)
        self.hidden_width = self.config_dict["hidden_layer_width"]
        self.netowrk_depth = self.config_dict["network_depth"]
        # model architecture
        self.layers = nn.Sequential(
            nn.Linear(in_features, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, out_features)
        )

    def forward(self, features: torch.tensor):
        """Defines forward pass of the model.

        Parameters
        ----------
        features : torch.tensor
            Tensor of model features.
        """
        return self.layers(features).reshape(-1)
        
# %%
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

def train(model, data_loaders: list, config_file: str):
    """ Trains model and prints accuracy statistics for training,
    testing, and validation sets.

    Parameters
    ----------
    model : Network
        Model initialised using Network class.
    data_loader : list
        List of train, test, and validation set DataLoaders.
    config_file : str
        File path of YAML file containing network parameters.
    """
    # get configuration parameters as a dictionary
    config_dict = get_nn_config(config_file)

    # get optimiser
    if config_dict["optimiser"] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=config_dict["learning_rate"])
    else:
        print("Invalid optimiser.")
        return
    
    # initialise tensorboard visualiser and metric variables
    writer = SummaryWriter()
    train_batch_index = 0
    test_batch_index = 0
    val_batch_index = 0
    min_val_loss = np.inf
    min_val_epoch = 0

    for epoch in range(config_dict["epochs"]):
        # training loop
        model.train()
        train_loss = 0.0
        train_r_squared = 0.0
        train_start_time = time.time()
        latency = np.zeros(len(data_loaders[0]))
        for i, (features, labels) in enumerate(data_loaders[0]): # train_loader
            # check if GPU is available
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()
            # forward pass
            prediction_start_time = time.time()
            prediction = model(features)
            prediction_end_time = time.time()
            latency[i] = prediction_end_time - prediction_start_time
            # calculate loss (mean squared error) and r_sqaured metric
            loss = F.mse_loss(prediction, labels)
            train_batch_r_squared = r2_score(labels.detach().numpy(), prediction.detach().numpy())
            # backpropagation
            loss.backward()
            # optimisation
            optimiser.step()
            optimiser.zero_grad() # clear gradients
            train_loss += loss.item()
            train_r_squared += train_batch_r_squared
            # add data to writer for tensorboard visualisation
            writer.add_scalar(tag=f"Train Loss", scalar_value=loss.item(), global_step=train_batch_index)
            train_batch_index += 1
        train_end_time = time.time()
        training_duration = train_end_time - train_start_time
        
        
        # set model to evaluation mode        
        model.eval()
        # test loop
        test_loss = 0.0
        test_r_squared = 0.0
        for features, labels in data_loaders[1]: # test_loader
            # check if GPU is available
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()
            # forward pass
            prediction = model(features)
            # calculate loss (mean squared error)
            loss = F.mse_loss(prediction, labels)
            test_batch_r_squared = r2_score(labels.detach().numpy(), prediction.detach().numpy())
            test_loss += loss.item()
            test_r_squared += test_batch_r_squared
            # add data to writer for tensorboard visualisation
            writer.add_scalar(tag=f"Test Loss", scalar_value=loss.item(), global_step=test_batch_index)
            test_batch_index += 1

        # validation loop          
        val_loss = 0.0
        val_r_squared = 0.0
        for features, labels in data_loaders[2]: # val_loader
            # check if GPU is available
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()
            # forward pass
            prediction = model(features)
            # calculate loss (mean squared error)
            loss = F.mse_loss(prediction, labels)
            val_batch_r_squared = r2_score(labels.detach().numpy(), prediction.detach().numpy())
            val_loss += loss.item()
            val_r_squared += val_batch_r_squared
            # add data to writer for tensorboard visualisation
            writer.add_scalar(tag=f"Validation Loss", scalar_value=loss.item(), global_step=val_batch_index)
            val_batch_index += 1

        # print epoch statistics
        if (epoch + 1) % 1 == 0:
            print(f"""Epoch: {epoch + 1}    Training Loss: {train_loss / len(loader_list[0]):.4f}, Training r_squared: {train_batch_r_squared / len(loader_list[0]):.4f}
            Test Loss: {test_loss / len(loader_list[1]):.4f}, Testing r_squared: {test_batch_r_squared / len(loader_list[1]):.4f}
            Validation Loss: {val_loss / len(loader_list[2]):.4f}, Validation r_squared: {val_batch_r_squared / len(loader_list[2]):.4f}\n""")

        if min_val_loss > val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch + 1
    
    # print overall statistics
    print(f"""\nLowest validation loss is {min_val_loss / len(loader_list[2]):.4f} at epoch {min_val_epoch}.\n""")

    metrics_dict = {
        "RMSE_loss" : {"train" : round(train_loss / len(loader_list[0]), 4), "test" : round(test_loss / len(loader_list[1]), 4), "validation" : round(val_loss / len(loader_list[2]), 4)},
        "R_squared" : {"train" : round(train_batch_r_squared / len(loader_list[0]), 4), "test" : round(test_batch_r_squared / len(loader_list[1]), 4), "validation" : round(val_batch_r_squared / len(loader_list[2]), 4)},
        "training_duration (s)" : round(training_duration, 3),
        "inference_latency" : round(latency.mean())
    }

    return metrics_dict

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
        
    # save hyperparameters in json file
    with open(f"{folder_path}/hyperparams.json", "w") as file:
        json.dump(hyperparams, file)

    # save model metrics in json file
    with open(f"{folder_path}/metrics.json", "w") as file:
        json.dump(metrics, file)
# %%
if __name__ == "__main__":
    # seed RNG for reproducability
    torch.manual_seed(42)

    # import AirBnB dataset, isolate and normalise numerical data and split into features and labels
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()
    feature_df_normalised, label_series = read_in_data()

    # create torch dataset
    dataset = AirBnBNightlyPriceImageDataset(feature_df_normalised, label_series)

    # split data into train, test, and validation sets
    train_subset, test_subset = random_split(dataset, [663, 166])
    test_subset, val_subset = random_split(test_subset, [132, 34])

    # create DataLoaders for each set
    BATCH_SIZE = 4
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    loader_list = [train_loader, test_loader, val_loader]

    # initiate and train model
    in_features = len(feature_df_normalised[0])
    out_features = 1
    config_path = "nn_config.yaml"

    nn_model = Network(in_features, out_features, config_path)
    metrics_dict = train(nn_model, loader_list, config_path)
    print(f"Metrics dictionary:\n{metrics_dict}.")

    # save model
    hyperparams_dict = get_nn_config("nn_config.yaml")
    save_model(nn_model, hyperparams_dict, metrics_dict)

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
        learning_rate = random.choice([1/i for i in [10**j for j in range(1, 5)]])
        hidden_layer_width = random.choice([i for i in [2**j for j in range(1, 5)]])
        epochs = random.choice([i for i in range(10, 50)])
        config_dict = {
            "optimiser" : "SGD",
            "learning_rate" : learning_rate,
            "hidden_layer_width" : hidden_layer_width,
            "network_depth" : None,
            "epochs" : epochs
        }
        # print(f"{config_dict}\n")
        dict_list.append(config_dict)

    return dict_list

def save_configs_as_yaml(config_list: list):
    for idx, config in enumerate(config_list):
        with open(f"network_configs/{idx}.yaml", "w") as file:
            yaml.dump(config, file)
# %%
my_dict = generate_nn_configs(2)

save_configs_as_yaml(my_dict)