# %%
from read_tabular_data import TabularData
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
# %%
if __name__ == "__main__":
    # load in data
    np.random.seed(20)
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()

    feature_df, label_series = tabular_df.load_airbnb(
        numerical_tabular_df,
        label="Price_Night"
    )
    feature_df = feature_df.drop("ID", axis=1)


    X_train, X_test, y_train, y_test = train_test_split(feature_df, 
                                                    label_series, 
                                                    test_size=0.3
                                                    )

    X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                                y_test,
                                                                test_size=0.3
                                                                )

    model = make_pipeline(StandardScaler(), SGDRegressor())
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    mse_loss_train = metrics.mean_squared_error(y_train, y_train_pred)
    mse_loss_test = metrics.mean_squared_error(y_test, y_test_pred)
    mse_loss_validation = metrics.mean_squared_error(y_validation, y_validation_pred)
    print("Mean Squared Error Loss on Training data: ", mse_loss_train)
    print("Mean Squared Error Loss on Test data: ", mse_loss_test)
    print("Mean Squared Error Loss on Validation data: ", mse_loss_validation)

    plt.figure()
    plt.scatter(np.arange(50), y_validation_pred[:50], c="r", label="Predictions")
    plt.scatter(np.arange(50), label_series[:50], c="b", label="True Labels", marker="x")
    plt.legend()
    plt.xlabel("Sample Numbers")
    plt.ylabel("Values")
    plt.show()

    print(f"Root mean squared error for training set: {mse_loss_train**0.5}")
    print(f"Root mean squared error for test set: {mse_loss_test**0.5}")

    r_squared_train = metrics.r2_score(y_train, y_train_pred)
    r_squared_test = metrics.r2_score(y_test, y_test_pred)
    print(f"R^2 for training set: {r_squared_train}")
    print(f"R^2 for test set: {r_squared_test}")

# %%