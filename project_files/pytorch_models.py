# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch
# %%
class AirbnbNightlyPriceImageDataset(Dataset):
    """Creates a PyTorch dataset  of the AirBnb data that returns a tuple 
    of (features, labels) when indexed.
    """
    def __init__(self, features, labels) -> None:
        super().__init__()
        assert len(features) == len(labels) # ensure features and label sets are the same length
        self.features = features
        self.labels = labels
    
    def __getitem__(self, index):
        return torch.tensor(self.features[index]), self.labels[index]
    
    def __len__(self):
        return len(self.features)
# %%
if __name__ == "__main__":
    # import AirBnb dataset, isolate numerical data and split it into features and labels
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()
    feature_df_scaled, label_series = read_in_data()
    
    # create custom torch dataset with the imported data
    dataset = AirbnbNightlyPriceImageDataset(feature_df_scaled, label_series)

    # split data into train, test, and validation sets
    train_subset, test_subset = random_split(dataset, [663, 166])
    test_subset, val_subset = random_split(test_subset, [132, 34])

    # create DataLoader to load in data for each set
    BATCH_SIZE = 10
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_subset, shuffle=True, batch_size=BATCH_SIZE)


# %%
