#%%
from read_tabular_data import TabularData
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#%%
if __name__ == "__main__":
    # load in data
    np.random.seed(42)
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

    model = SGDRegressor()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    mse_loss = mean_squared_error(y_validation, y_validation_pred)
    print("Mean Squared Error Loss on Validation data: ", mse_loss)

    plt.figure()
    plt.scatter(np.arange(50), y_validation_pred[:50], c="r", label="Predictions")
    plt.scatter(np.arange(50), label_series[:50], c="b", label="True Labels", marker="x")
    plt.legend()
    plt.xlabel("Sample Numbers")
    plt.ylabel("Values")
    plt.show()

#%%


