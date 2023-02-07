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
