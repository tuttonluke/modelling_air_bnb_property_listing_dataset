# %%
from new_neural_net import visualise_features_vs_target, AirBnBNightlyPriceImageDataset, generate_nn_configs, save_configs_as_yaml, train_networks, find_best_nn
from regression_modelling import read_in_data
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
import torch

# %%
if __name__ == "__main__":
    # seed RNG for reproducability
    np.random.seed(42)
    torch.manual_seed(42)

    # import AirBnB dataset, isolate and normalise numerical data and split into features and labels
    feature_df_normalised, label_series, feature_names = read_in_data(label="beds")
    
    # visualise data
    visualise_features_vs_target(feature_df_normalised, label_series, feature_names, target="# Bedrooms")

    # create torch dataset and split into train and test subsets
    dataset = AirBnBNightlyPriceImageDataset(feature_df_normalised, label_series)
    train_subset, test_subset = random_split(dataset, [729, 100])  

    # initialise DataLoaders
    BATCH_SIZE = 4
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)
    
    # generate and save a number of hyperparameter configurations
    configurations_dict = generate_nn_configs(n_configs=16)
    save_configs_as_yaml(configurations_dict)

    # initialise, train, and evaluate models
    train_networks(train_loader, test_loader, in_features=11, hidden_width=128, out_features=1)

    # find the best model from all those trained
    find_best_nn()

# TODO
# sort out files!
# get feature names docstring
# consolidate if __name__ section into a function