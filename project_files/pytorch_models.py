# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from torch.utils.data import Dataset
import torch
# %%
class AirbnbNightlyPriceImageDataset(Dataset):
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
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()

    feature_df_scaled, label_series = read_in_data()

    dataset = AirbnbNightlyPriceImageDataset(feature_df_scaled, label_series)

# %%
