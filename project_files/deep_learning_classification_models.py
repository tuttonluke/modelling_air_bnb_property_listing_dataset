# %%
from deep_learning_regression_models import AirBnBNightlyPriceImageDataset, NeuralNetwork
from torch.utils.data import  DataLoader, random_split
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import winsound
import numpy as np
from utils.data_handling_utils import read_in_data
import seaborn as sns
from utils.read_tabular_data import TabularData
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from utils.nn_utils import get_nn_config, generate_nn_configs, find_best_classification_nn
from utils.nn_utils import save_configs_as_yaml, save_model
from sklearn.metrics import f1_score
import os
from utils.visualisation_utils import visualise_classification_metrics
# %%
def accuracy_test(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_true).float()
    accuracy = correct_pred.sum() / len(correct_pred)
    accuracy = torch.round(accuracy*100)

    return accuracy

def evaluate_test(model, test_loader: DataLoader):
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            y_test_pred = model(X_batch)
            y_true.append(np.array(y_batch))
            y_batch, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred.append(y_pred_tags.numpy())
    y_pred = [a.squeeze().tolist() for a in y_pred]

    return y_pred, y_true


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, config_dict: dict) -> tuple:
    """Trains the model.

    Parameters
    ----------
    model : NeuralNetwork
        Neural network model.
    train_loader : DataLoader
        DataLoader of training data.
    val_loader : DataLoader
        DataLoader of validation data.
    config_dict : dict
        Dictionary of configuration hyperparameters.

    Returns
    -------
    tuple
        Training time in seconds (float), and dictionaries of accuracy and loss statistics.
    """
    # get hyperparameters
    learning_rate = config_dict["learning_rate"]
    epochs = config_dict["epochs"]

    # define optimiser, criterion, and writer for tensorboard visualisation
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    train_batch_index = 0
    val_batch_index = 0
    train_start_time = time.time()

    accuracy_stats = {
    'train': [],
    "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    for epoch in range(epochs):
        # TRAINING
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        model.train()
        for X, y in train_loader:
            optimiser.zero_grad()
            y_train_pred = model(X)
            train_loss = loss_function(y_train_pred, y)
            train_acc = accuracy_test(y_train_pred, y)
            train_loss.backward()
            optimiser.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            writer.add_scalar(tag="Train Loss", scalar_value=train_loss.item(), global_step=train_batch_index)
            train_batch_index += 1

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0.0
            val_epoch_acc = 0.0

            model.eval()
            for X, y in val_loader:
                y_val_pred = model(X)
                val_loss = loss_function(y_val_pred, y)
                val_acc = accuracy_test(y_val_pred, y)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                writer.add_scalar(tag="Validation Loss", scalar_value=val_loss.item(), global_step=val_batch_index)
                val_batch_index += 1

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.2} | Val Loss: {val_epoch_loss/len(val_loader):.2} | Train Acc: {train_epoch_acc/len(train_loader):.3}| Val Acc: {val_epoch_acc/len(val_loader):.3}')

    train_end_time = time.time()
    training_duration = train_end_time - train_start_time

    print(f"Training duration: {training_duration:.2} seconds.")

    return training_duration, accuracy_stats, loss_stats

def train_networks(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, in_features: int, hidden_width: int, out_features: int):
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
        "f1_macro" : None,
        "training_time" : None
    }
    # locate and loop through YAML configureation files
    for root, dirs, files in os.walk(config_directory):
        for file in files:
            nn_model = NeuralNetwork(in_features, hidden_width, out_features)
            config_dict = get_nn_config(file)
            training_time, accuracy_stats, loss_stats = train_model(nn_model, train_loader, val_loader, config_dict)
            
            y_test_pred, y_test_true = evaluate_test(nn_model, test_loader)
            f1_macro = f1_score(y_test_true, y_test_pred, average="macro")
            metrics_dict["f1_macro"] = f1_macro.astype("float64")
            metrics_dict["training_time"] = training_time
            # print parameters and metrics
            print(config_dict)
            print(f"F1 Score (macro): {metrics_dict['f1_macro']:.4f}")
            print(f"Training duration: {training_time:.2f} seconds.\n")
            
            fig1, fig2 = visualise_classification_metrics(accuracy_stats, loss_stats, y_test_pred, y_test_true)

            # save model
            save_model(nn_model, config_dict, metrics_dict, fig1, fig2, model_type="classification")

def evaluate_best_model(label="beds", n_configs=16):
    """Loads, preprocesses, and splits data based on the target label given.
    The data is then visualised, and a numner of neural network models trained.
    The models are saved along with their hyperparameters and evaluation metrics,
    and the best model is highlighted.

    Parameters
    ----------
    label : str, optional
        Name of target label, by default "beds"
    n_configs : int, optional
        Number of model configurations to evaluate, by default 16
    """
    # seed RNG for reproducability
    np.random.seed(42)
    torch.manual_seed(42)

    # import AirBnB dataset, isolate and normalise numerical data and split into features and labels
    feature_array_normalised, label_array, feature_names = read_in_data(label=label)
    label_array = class_to_index(label_array)

    # create torch dataset and split into train and test subsets
    dataset = AirBnBNightlyPriceImageDataset(feature_array_normalised, label_array, model_type="classification")
    train_subset, test_subset = random_split(dataset, [650, 179])
    test_subset, val_subset = random_split(test_subset, [100, 79])

    # dataloaders
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=4)
    test_loader = DataLoader(test_subset, batch_size=1)
    val_loader = DataLoader(val_subset, batch_size=1)
    
    # generate and save a number of hyperparameter configurations
    configurations_dict = generate_nn_configs(n_configs=n_configs)
    save_configs_as_yaml(configurations_dict)

    # initialise, train, and evaluate models
    train_networks(train_loader, val_loader, test_loader, in_features=11, hidden_width=128, out_features=17)

    # find the best model from all those trained
    find_best_classification_nn()

def class_to_index(array):
    for index, value in enumerate(array):
        array[index] = int(value - 1)
    
    return array
# %%
if __name__ == "__main__":

    # # import AirBnB dataset, isolate and normalise numerical data and split into features and labels
    # df = TabularData()
    # num_df = df.get_numerical_data_df()
    # ax = sns.countplot(x = "beds", data=num_df)
    # ax.set_title("Class Distribution")
    # ax.set_xlabel("Number of Beds")
    # ax.set_ylabel("Count")




    evaluate_best_model(label="beds", n_configs=2)

    # make a sound when code has finished running
    duration = 1000 # milliseconds
    frequency = 440 # Hz
    winsound.Beep(frequency, duration)

# %%

# TODO


# move helper functions to utils
# typing, docstrings, comments
# confusion matrix visuals
