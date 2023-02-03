# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
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


        