# %%
import matplotlib.pyplot as plt
import numpy as np
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