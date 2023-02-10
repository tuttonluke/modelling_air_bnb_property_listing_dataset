# %%
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.nn_utils import get_nn_config, generate_nn_configs, find_best_nn
from utils.nn_utils import save_configs_as_yaml, save_model
from utils.data_handling_utils import read_in_data
from utils.visualisation_utils import visualise_features_vs_target
import itertools
import numpy as np
import os
import time
import torch
import torch.nn as nn
import winsound
# %% Dataset and Neural Network classes

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

# %% Neural Network training and evaluation functions

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
    train_start_time = time.time()

    # training loop
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

    # time training
    train_end_time = time.time()
    training_duration = train_end_time - train_start_time

    return training_duration

def train_networks(train_loader: DataLoader, test_loader: DataLoader, in_features: int, hidden_width: int, out_features: int):
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
        loader : DataLoader
            Training set dataloader.
    """
    config_directory = "network_configs"
    metrics_dict = {
        "test_MSE" : None,
        "test_r_squared" : None,
        "training_time" : None
    }
    # locate and loop through YAML configureation files
    for root, dirs, files in os.walk(config_directory):
        for file in files:
            nn_model = NeuralNetwork(in_features, hidden_width, out_features)
            config_dict = get_nn_config(file)
            training_time = train_model(nn_model, train_loader, config_dict)
            mse, r2 = evaluate_model(nn_model, test_loader)
            metrics_dict["test_MSE"] = mse.astype("float64")
            metrics_dict["test_r_squared"] = r2.astype("float64")
            metrics_dict["training_time"] = training_time
            # print parameters and metrics
            print(config_dict)
            print(f"Test MSE: {metrics_dict['test_MSE']:.2f}")
            print(f"Test r_squared score: {metrics_dict['test_r_squared']:.4f}")
            print(f"Training duration: {training_time:.2f} seconds.\n")
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

# %% Main function

def evaluate_best_model(label="Price_Night", n_configs=16):
    """Loads, preprocesses, and splits data based on the target label given.
    The data is then visualised, and a numner of neural network models trained.
    The models are saved along with their hyperparameters and evaluation metrics,
    and the best model is highlighted.

    Parameters
    ----------
    label : str, optional
        Name of target label, by default "Price_Night"
    n_configs : int, optional
        Number of model configurations to evaluate, by default 16
    """
    # seed RNG for reproducability
    np.random.seed(42)
    torch.manual_seed(42)

    # import AirBnB dataset, isolate and normalise numerical data and split into features and labels
    feature_df_normalised, label_series, feature_names = read_in_data(label=label)

    # visualise data
    visualise_features_vs_target(feature_df_normalised, label_series, feature_names, target=label)

    # create torch dataset and split into train and test subsets
    dataset = AirBnBNightlyPriceImageDataset(feature_df_normalised, label_series)
    train_subset, test_subset = random_split(dataset, [729, 100])

    # initialise DataLoaders
    BATCH_SIZE = 4
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)
    
    # generate and save a number of hyperparameter configurations
    configurations_dict = generate_nn_configs(n_configs=n_configs)
    save_configs_as_yaml(configurations_dict)

    # initialise, train, and evaluate models
    train_networks(train_loader, test_loader, in_features=11, hidden_width=128, out_features=1)

    # find the best model from all those trained
    find_best_nn()
# %%
if __name__ == "__main__":
    
    # evaluate best model with Price_Night as the target label
    evaluate_best_model(label="Price_Night", n_configs=16)

    # make a sound when code has finished running
    duration = 1000 # milliseconds
    frequency = 440 # Hz
    winsound.Beep(frequency, duration)