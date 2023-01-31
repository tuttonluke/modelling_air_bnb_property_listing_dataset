# %%
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
