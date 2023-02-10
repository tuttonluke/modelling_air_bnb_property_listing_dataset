#%%
import pandas as pd
#%%
class TabularData:
    def __init__(self) -> None:
        self.data_file_path = "utils/tabular_data/clean_tabular_data.csv"
        self.tabular_df = pd.read_csv(self.data_file_path)
    
    def get_numerical_data_df(self) -> pd.DataFrame:
        """Removes all non-numerical data from clean_tabular_data DataFrame,
        and returns the new DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with only numerical data.
        """
        self.tabular_df = self.tabular_df.drop([
                                                    "ID",
                                                    "Unnamed: 0",
                                                    "Category",
                                                    "Title",
                                                    "Description",
                                                    "Amenities",
                                                    "Location",
                                                    "url"
                                                    ], axis=1)
        return self.tabular_df
    
    def load_airbnb(self, df: pd.DataFrame, label: str="Price_Night") -> tuple:
        """Splits tabular data into a DataFrame of Features and a Series of labels.

        Parameters
        ----------
        df : pd.DataFrame
        label : str, optional
            The column to be predicted, by default "Price_Night"

        Returns
        -------
        tuple
            Tuple of DataFrame of features, Series of labels.
        """
        label_series = df[label]
        feature_df = df.drop(label, axis=1)

        return feature_df, label_series
    
    def get_classification_data(self) -> pd.DataFrame:
        """Removes all non-numerical data from clean_tabular_data DataFrame,
        except the label Category and returns the new DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with only numerical data.
        """
        self.tabular_df = self.tabular_df.drop([
                                                    "ID",
                                                    "Unnamed: 0",
                                                    "Title",
                                                    "Description",
                                                    "Amenities",
                                                    "Location",
                                                    "url"
                                                    ], axis=1)
        return self.tabular_df
    
    def get_feature_names(self, label: str="Price_Night") -> list:
        """Returns a list of column names in self.tabular_df,
        excluding the one marked as the label.

        Parameters
        ----------
        label : str, optional
            Label column name, by default "Price_Night".

        Returns
        -------
        list
            List of feature names.
        """
        feature_list = self.tabular_df.columns.values.tolist()
        feature_list.remove(label)

        return feature_list

#%%
if __name__ == "__main__":
    clean_tabular_df = TabularData()
    clean_numerical_tabular_df = clean_tabular_df.get_numerical_data_df()

    feature_df, label_series = clean_tabular_df.load_airbnb(
        clean_numerical_tabular_df,
        label="Price_Night"
    )

    feature_list = clean_tabular_df.get_feature_names()
    print(feature_list)
