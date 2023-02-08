# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from sklearn.metrics import mean_squared_error, r2_score
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
        """
        Parameters
        ----------
        in_features : int
            Number of features.
        hidden_width : int
            Number of hidden layer nodes.
        out_features : int
            Number of labels.
        """
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
def train_model(model, loader: DataLoader, config_dict: dict):
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

def train_networks(in_features: int, hidden_width: int, out_features: int):
    """Trains models based on yaml configuration files stored in the
    config_dictionary file location. The model is then saved, and the
    metrics and hyperarameters are saved in .json files.

    Parameters
    ----------
        in_features : int
            Number of features.
        hidden_width : int
            Number of hidden layer nodes.
        out_features : int
            Number of labels.
    """
    config_directory = "network_configs"
    metrics_dict = {
        "test_MSE" : None,
        "test_r_squared" : None
    }
    # locate and loop through YAML configureation files
    for root, dirs, files in os.walk(config_directory):
        for file in files:
            nn_model = NeuralNetwork(in_features, hidden_width, out_features)
            config_dict = get_nn_config(file)
            train_model(nn_model, train_loader, config_dict)
            mse, r2 = evaluate_model(nn_model, test_loader)
            metrics_dict["test_MSE"] = mse.astype("float64")
            metrics_dict["test_r_squared"] = r2.astype("float64")
            
            # print parameters and metrics
            print(config_dict)
            print(f"Test MSE: {metrics_dict['test_MSE']:.2f}")
            print(f"Test r_squared score: {metrics_dict['test_r_squared']:.4f}\n")

            # save model
            save_model(nn_model, config_dict, metrics_dict)

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
        learning_rate = random.choice([1/i for i in [10**j for j in range(3, 5)]])
        epochs = random.choice([i for i in range(5, 30)])
        config_dict = {
            "learning_rate" : learning_rate,
            "epochs" : epochs
        }
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
    
    # generate and save a number of hyperparameter configurations
    configurations_dict = generate_nn_configs(n_configs=3)
    save_configs_as_yaml(configurations_dict)

    # initialise, train, and evaluate models
    train_networks(in_features=11, hidden_width=128, out_features=1)


# TODO
# train time
# prediction time
# %%
def find_best_nn():
    pass