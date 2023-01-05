#%%
from read_tabular_data import TabularData
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
#%%
if __name__ == "__main__":
    # load in data
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()

    feature_df, label_series = tabular_df.load_airbnb(
        numerical_tabular_df,
        label="Price_Night"
    )
    feature_df = feature_df.drop("ID", axis=1)

    #
    model = LinearRegression()
    model.fit(feature_df, label_series)
    y_pred = model.predict(feature_df)

    plt.figure()
    plt.scatter(np.arange(50), y_pred[:50], c="r", label="Predictions")
    plt.scatter(np.arange(50), label_series[:50], c="b", label="True Labels", marker="x")
    plt.legend()
    plt.xlabel("Sample Numbers")
    plt.ylabel("Values")
    plt.show()

