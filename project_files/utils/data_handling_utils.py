# %%
from utils.read_tabular_data import TabularData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
# %%
def read_in_data(label: str="Price_Night") -> tuple:
    """Reads in, cleans, splits, and normalises data for analysis.

    Returns
    -------
    tuple
        Tuple containing numpy arrays of features and labels,
        as well as a list of feature names.
    """
    tabular_df = TabularData()
    numerical_tabular_df = tabular_df.get_numerical_data_df()

    feature_df, label_series = tabular_df.load_airbnb(
        numerical_tabular_df,
        label=label
    )
    feature_df_scaled = normalise_data(feature_df)
    feature_names = tabular_df.get_feature_names(label=label)

    return feature_df_scaled, np.array(label_series), feature_names

def split_data(feature_dataframe: pd.DataFrame, label_series:pd.Series, test_size: float=0.3) -> tuple:
    """Splits feature dataframe into train, test, and validation sets
    in a proportion of test_size.

    Parameters
    ----------
    feature_dataframe : pd.DataFrame
        DataFrame of model features.
    label_series : pd.Series
        Series of model labels.
    test_size : float, optional
        proportion of data to split into test set and validation set, by default 0.3

    Returns
    -------
    tuple
        tuple of arrays of train, test, and validation data.
    """
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(feature_dataframe, 
                                                    label_series, 
                                                    test_size=test_size
                                                    )
    X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                            y_test,
                                                            test_size=test_size
                                                            )                                                    
    return X_train, y_train, X_test, y_test, X_validation, y_validation

def normalise_data(data: pd.DataFrame) -> np.array:
    """
    Normalises input data (DataFrame) with Min-Max scaling.

    Parameters
    ----------
    data : pd.Data
        Features data.

    Returns
    -------
    np.array
        Scaled features data in numpy array format.
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.to_numpy())

    return  data_scaled