# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, features: torch.tensor, labels: torch.tensor) -> None:
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
# %%
def train(model, loader, learning_rate, epochs, ):
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
            loss = loss_function(outputs, labels)
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
        print("Epoch [%d]/[%d] running accumulative loss across all batches: %.3f" %
                    (epoch + 1, epochs, running_loss))
        running_loss = 0.0

# %% HELPER FUNCTIONS
def visualise_data(X, y, feature_names):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
    for i, (ax, col) in enumerate(zip(axs.flat, feature_names)):
        x = X[:, i]
        pf = np.polyfit(x, y, 1)
        p = np.poly1d(pf)

        ax.plot(x, y, "o")
        ax.plot(x, p(x), "r--")

        ax.set_title(col + " vs Price per Night")
        ax.set_xlabel(col)
        ax.set_ylabel("Price per Night")
    plt.show()

# %%
if __name__ == "__main__":
    # seed RNG for reproducability
    np.random.seed(42)
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

    # initialise DataLoaders
    batch_size = 4
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    feature_names = tabular_df.get_feature_names()
    print(feature_names)
    visualise_data(feature_df_normalised, label_series, feature_names)

