# %%
from read_tabular_data import TabularData
from regression_modelling import read_in_data
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import yaml
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
        return torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)

class LinearRegression(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features)
    
    def forward(self, features):
        return self.linear_layer(features).reshape(-1) # make prediction

class NN(torch.nn.Module):
    def __init__(self, in_features, out_features, config_dict) -> None:
        super().__init__()
        self.config_dict = get_nn_config(config_dict)
        self.hidden_width = self.config_dict["hidden_layer_width"]
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, self.hidden_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_width, out_features)
        )
    
    def forward(self, features):
        return self.layers(features).reshape(-1)

def train(model, data_loader, tag: str, epochs=10):

    # initialise optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    # initialise tensorboard visualiser
    writer = SummaryWriter()
    batch_index = 0

    for epoch in range(epochs):
        for batch in data_loader:
            features, labels = batch
            prediction = model(features)
            # calculate loss
            loss = F.mse_loss(prediction, labels)
            # backpropagation (populate gradients)
            loss.backward()
            print(loss)
            # optimisation
            optimiser.step()
            optimiser.zero_grad() # reset gradients
            # add data to writer for visualisation
            writer.add_scalar(tag=f"{tag} Loss", scalar_value=loss.item(), global_step=batch_index)
            batch_index += 1

def get_nn_config(file_path: str) -> dict:
    with open(file_path, "r") as file:
        try:
            config_dict = yaml.safe_load(file)
            print(config_dict)
        except yaml.YAMLError as e:
            print(e)
    return config_dict
    
def train_with_config(model, data_loader, tag: str, config_file: str):
    # get config as dictionary
    config_dict = get_nn_config(config_file)

    # get optimiser
    if config_dict["optimiser"] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=config_dict["learning_rate"])
    else:
        print("Invalid optimiser.")
    # initialise tensorboard visualiser
    writer = SummaryWriter()
    batch_index = 0

    # train the model
    for epoch in range(config_dict["epochs"]):
        for batch in data_loader:
            features, labels = batch
            prediction = model(features)
            # calculate loss
            loss = F.mse_loss(prediction, labels)
            # backpropagation
            loss.backward()
            print(loss)
            # optimisation
            optimiser.step()
            optimiser.zero_grad() # reset gradients
            # add data to writer for visualisation
            writer.add_scalar(tag=f"{tag} Loss", scalar_value=loss.item(), global_step=batch_index)
            batch_index += 1
    
# %%
if __name__ == "__main__":
    # seed RNG for reproducability
    torch.manual_seed(42)
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
    BATCH_SIZE = 4
    train_loader = DataLoader(train_subset, shuffle=False, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_subset, shuffle=True, batch_size=BATCH_SIZE)

    in_features = len(feature_df_scaled[0]) # number of features (11)
    out_features = 1 # number of labels

    # # initiate and train Linear Regression model
    # lr_model = LinearRegression(in_features, out_features)
    
    # train(lr_model, train_loader, "Train")
    # train(lr_model, val_loader, "Validation")

    # # initiate and train neural network model
    # nn_model = NN(in_features, out_features, "nn_config.yaml")
    
    # train(nn_model, train_loader, "Train")
    # train(nn_model, val_loader, "Validation")

    config_path = "nn_config.yaml"
    nn_model = NN(in_features, out_features, config_path)
    train_with_config(nn_model, train_loader, "Train", config_path) # training set  
# %%

    





