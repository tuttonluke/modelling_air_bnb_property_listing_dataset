# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import datetime
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()
        # assert feature and label sets are the same length
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels
    
    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)

    def __len__(self):
        return len(self.features)

class NeuralNetwork(nn.Module):
    """Defines the neural network architecture for the regression problem.
    """
    def __init__(self, in_features: int, hidden_width: int, out_features: int) -> None:
        super().__init__()

        # model architecture
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, out_features)
        )
    
    def forward(self, features: torch.tensor):
        return self.layers(features)
# %% TRAIN AND EVALUATION FUNCTIONS
def train(model, loader: DataLoader, config_dict: dict):
    """Trains the model.

    Parameters
    ----------
    model : NeuralNetwork
        Neural network model.
    loader : DataLoader
        DataLoader of training data.
    config_dict : dict
        Dictionary of configuration hyperparameters.
    """
    # get hyperparameters
    learning_rate = config_dict["learning_rate"]
    epochs = config_dict["epochs"]
    
    # define optimiser, criterion, and writer for tensorboard visualisation
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    writer = SummaryWriter()
    batch_index = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in loader:
            # forward pass
            outputs = model(features)
            # define loss
            loss = loss_function(outputs, labels.unsqueeze(-1))
            # zero gradients
            optimiser.zero_grad()
            # compute gradients
            loss.backward()
            # accumulate running loss
            running_loss += loss.item()
            # update weights 
            optimiser.step()
            # add to writer for tensorboard visualisaiton
            writer.add_scalar(tag="Train Loss", scalar_value=loss.item(), global_step=batch_index)
            batch_index += 1
        # print epoch loss
        if epoch % 10 == 0:
            print("Epoch [%d]/[%d] running accumulative loss across all batches: %.3f" %
                        (epoch + 1, epochs, running_loss))
        running_loss = 0.0

def evaluate_model(model, loader: DataLoader) -> tuple:
    """Calculate mean squared arro (MSE) and r^2 score
    of the model for data in the loader.

    Parameters
    ----------
    model : NeuralNetwork
        Neural Network model.
    loader : DataLoader
        DataLoader of test (evaluation) data.

    Returns
    -------
    tuple
        tuple of MSE and r^2 values.
    """
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            predicted = list(itertools.chain(*np.array(outputs)))
            y_pred.append(predicted)
            y_true.append(np.array(y))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2
# %% HELPER FUNCTIONS
def visualise_features_vs_target(X: np.ndarray, y: np.ndarray, feature_names: list):
    """Creates a 3x4 plot which visualises each feature seperately against
    the target label as a scatter plot. Also plots a line fitted to minimise
    the squared error in each case.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Label array.
    feature_names : list
        List of feature names for use in subplot titles.
    """
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
    for i, (ax, col) in enumerate(zip(axs.flat, feature_names)):
        x = X[:, i]
        # calculate line of best fit
        pf = np.polyfit(x, y, 1)
        p = np.poly1d(pf)

        ax.plot(x, y, "o")
        ax.plot(x, p(x), "r--")

        fig.suptitle("Visualisation of Each Feature vs Target Label", y=0.93, size=24)
        ax.set_title(col + " vs Price per Night", size=16)
        ax.set_xlabel(col, size=14)
        if i % 4 == 0:
            ax.set_ylabel("Price per Night", size=14)
    plt.show()

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
        hidden_layer_width = random.choice([i for i in [2**j for j in range(3, 9)]])
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
# %%
if __name__ == "__main__":
    # seed RNG for reproducability
    np.random.seed(42)
    torch.manual_seed(42)

    # import AirBnB dataset, isolate and normalise numerical data and split into features and labels
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()
    feature_df_normalised, label_series = read_in_data()

    # Visualise Data
    feature_names = ["# Guests", "# Beds", "# Bathrooms", "Cleanliness Rating",
                    "Accuracy Rating", "Communication Rating", "Location Rating",
                    "Check-in Rating", "Value Rating", "Amenities Count", "# Bedrooms"]
    visualise_features_vs_target(feature_df_normalised, label_series, feature_names)

    # create torch dataset and split into train and test subsets
    dataset = AirBnBNightlyPriceImageDataset(feature_df_normalised, label_series)
    train_subset, test_subset = random_split(dataset, [729, 100])


    # initialise DataLoaders
    BATCH_SIZE = 4
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)

    
    # # generate and save a number of hyperparameter configurations
    # configurations_dict = generate_nn_configs(n_configs=6)
    # save_configs_as_yaml(configurations_dict)

    config_dict = {
        "learning_rate" : 0.001,
        "epochs" : 20
    }
    # initialise and train model
    model = NeuralNetwork(in_features=11, hidden_width=128, out_features=1)
    train(model, train_loader, config_dict)

    # evaluate test and validation metrics
    test_MSE, test_r2 = evaluate_model(model, test_loader)
    print(f"Test set MSE: {test_MSE:.2f}, Test r2_score: {test_r2:.4f}")


# %%
# TODO
# train and evaluate model docstrings
# all other helper functions