# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# %%
def visualise_features_vs_target(X: np.ndarray, y: np.ndarray, feature_names: list, target: str):
    """Creates a 3x4 plot which visualises each feature seperately against
    the target label as a scatter plot. Also plots a line fitted to minimise
    the squared error in each case.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Label array.
    feature_names : list
        List of feature names for use in subplot titles.
    target: str
        Name of target.
    """
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
    for i, (ax, col) in enumerate(zip(axs.flat, feature_names)):
        x = X[:, i]
        # calculate line of best fit
        pf = np.polyfit(x, y, 1)
        p = np.poly1d(pf)

        ax.plot(x, y, "o")
        ax.plot(x, p(x), "r--")

        fig.suptitle("Visualisation of Each Feature vs Target Label", y=0.93, size=24)
        ax.set_title(col + f" vs {target}", size=16)
        ax.set_xlabel(col, size=14)
        if i % 4 == 0:
            ax.set_ylabel(f"{target}", size=14)
    plt.show()

def visualise_classification_metrics(accuracy_stats, loss_stats, y_test_pred, y_test_true):
    # Create dataframes of data for plotting
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    
    # Plot accuracy and loss vs epoch
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,7))

    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=ax1)
    ax1.legend(fontsize=14, title_fontsize=16)
    ax1.set_title("Train and Validation Accuracy vs Epoch", fontsize=18)
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=ax2)
    ax2.legend(fontsize=14, title_fontsize=16)
    ax2.set_title("Train and Validation Loss vs Epoch", fontsize=18)
    ax2.set_xlabel("Epoch", fontsize=16)
    ax2.set_ylabel("Loss", fontsize=16)

    # plot the confusion matrix
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test_true, y_test_pred))
    
    fig2, ax3 = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(confusion_matrix_df, annot=True)
    ax3.set_title("Confusion Matrix", fontsize=14)
    ax3.set_xlabel("True Label", fontsize=12)
    ax3.set_ylabel("Predicted Label", fontsize=12)

    return fig, fig2
# %%
