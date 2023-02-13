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


def train(model, train_loader: DataLoader, val_loader: DataLoader, config_dict: dict):
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

        print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

    train_end_time = time.time()
    training_duration = train_end_time - train_start_time

    print(f"Training duration: {training_duration:.2} seconds.")

    return accuracy_stats, loss_stats
# %%
def class_to_index(array):
    for index, value in enumerate(array):
        array[index] = int(value - 1)
    
    return array
# %%
if __name__ == "__main__":

    # seed RNG for reproducability
    np.random.seed(42)
    torch.manual_seed(42)

    # import AirBnB dataset, isolate and normalise numerical data and split into features and labels
    df = TabularData()
    num_df = df.get_numerical_data_df()
    ax = sns.countplot(x = "beds", data=num_df)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Number of Beds")
    ax.set_ylabel("Count")

    feature_array_normalised, label_array, feature_names = read_in_data(label="beds")
    label_array = class_to_index(label_array)

    # create torch dataset and split into train and test subsets
    dataset = AirBnBNightlyPriceImageDataset(feature_array_normalised, label_array)
    train_subset, test_subset = random_split(dataset, [650, 179])
    test_subset, val_subset = random_split(test_subset, [100, 79])

    # dataloaders
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=4)
    test_loader = DataLoader(test_subset, batch_size=1)
    val_loader = DataLoader(val_subset, batch_size=1)

    # model
    config_dict = {
        "learning_rate" : 0.001,
        "epochs" : 10
    }
    model = NeuralNetwork(in_features=11, hidden_width=128, out_features=17)
    accuracy_stats, loss_stats = train(model, train_loader, val_loader, config_dict)

    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

    #
    y_test_pred, y_test_true = evaluate_test(model, test_loader)
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test_true, y_test_pred))#.rename(columns=index_to_class, index=index_to_class)

    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(confusion_matrix_df, annot=True)
    print(classification_report(y_test_true, y_test_pred))
    # make a sound when code has finished running
    duration = 1000 # milliseconds
    frequency = 440 # Hz
    winsound.Beep(frequency, duration)

# %%

