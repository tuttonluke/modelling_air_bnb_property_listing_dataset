# %%
from deep_learning_regression_models import AirBnBNightlyPriceImageDataset, NeuralNetwork
from torch.utils.data import  DataLoader, random_split
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import winsound
import numpy as np
# %%
def accuracy_test(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_true).float()
    accuracy = correct_pred.sum() / len(correct_pred)
    accuracy = torch.round(accuracy*100)

    return accuracy


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
    for epoch in epochs:
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
# %%
if __name__ == "__main":

    # seed RNG for reproducability
    np.random.seed(42)
    torch.manual_seed(42)

    # make a sound when code has finished running
    duration = 1000 # milliseconds
    frequency = 440 # Hz
    winsound.Beep(frequency, duration)