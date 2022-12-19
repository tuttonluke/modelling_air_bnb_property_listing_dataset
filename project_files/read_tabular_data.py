#%%
import pandas as pd
#%%
class TabularData:
    def __init__(self) -> None:
        self.data_file_path = "tabular_data/clean_tabular_data.csv"
        self.tabular_df = pd.read_csv(self.data_file_path)
    
    def get_numerical_data_df(self):
        """Removes all non-numerical data from clean_tabular_data DataFrame,
        and returns the new DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with only numerical data.
        """
        numerical_tabular_df = self.tabular_df.drop([
                                                    "Unnamed: 0",
                                                    "Category",
                                                    "Title",
                                                    "Description",
                                                    "Amenities",
                                                    "Location",
                                                    "url"
                                                    ], axis=1)
        return numerical_tabular_df
    
    def load_airbnb(self, df, label="Price_Night"):
        """_summary_

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
        

#%%
if __name__ == "__main__":
    clean_tabular_df = TabularData()
    clean_numerical_tabular_df = clean_tabular_df.get_numerical_data_df()

    feature_df, label_series = clean_tabular_df.load_airbnb(
        clean_numerical_tabular_df,
        label="Price_Night"
    )
