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

    def forward(self, features):
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
    with open(file_path, "r") as file:
        try:
            config_dict = yaml.safe_load(file)
            print(config_dict)
        except yaml.YAMLError as error:
            print(error)
    return config_dict
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

    # initiate model and training
    in_features = len(feature_df_normalised[0])
    out_features = 1
    config_path = "nn_config.yaml"

    nn_model = Network(in_features, out_features, config_path)
    
