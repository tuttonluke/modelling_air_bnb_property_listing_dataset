#%%
import pandas as pd
import missingno as msno
from ast import literal_eval
#%%
class AirBnbDataPreparation:
    def __init__(self) -> None:
        self.data_file_path = "tabular_data/listing.csv"
        self.listing_data = pd.read_csv(self.data_file_path)

    def initial_cleaning(self):
        """Perform simple initial cleaning tasks.
        """
        # Change categorical dtypes to category
        self.listing_data["Category"] = self.listing_data["Category"].astype("category")
        # drop bogus row
        self.listing_data = self.listing_data.drop(index=586)
        # Remove empty column
        self.listing_data = self.listing_data.drop("Unnamed: 19", axis=1)
        # reset indexes
        self.listing_data.reset_index(drop=True, inplace=True)
    

    def remove_rows_with_missing_ratings(self) -> pd.DataFrame:
        """Removes rows from the self.listing_data that have missing ratings data,
        or missing description.
        """
        self.listing_data.dropna(subset=[
                                        "Description",
                                        "Cleanliness_rating", 
                                        "Accuracy_rating",
                                        "Communication_rating",
                                        "Location_rating",
                                        "Check-in_rating",
                                        "Value_rating"
                                        ], inplace=True)
        self.listing_data.reset_index(drop=True, inplace=True)
    
    def clean_description_strings(self):
        """_summary_
        """
        # change string to list
        for index in range(len(self.listing_data["Description"])):
            self.listing_data["Description"][index] = literal_eval(self.listing_data["Description"][index]) 

        for index, description in enumerate(self.listing_data["Description"]):
            for description_index, description_element in enumerate(description):
                if description_element in ["About this space", 
                                            "The space", 
                                            "Guest access", 
                                            "Other things to note",
                                            "",
                                            " "
                                            ]:
                    self.listing_data["Description"][index].pop(description_index)
    
    def set_default_feature_values(self):
        """Replaces NaN values in the "guests", "beds", "bathrooms", 
        and "bedrooms" columns with value specified (default 1).
        """
        self.listing_data = self.listing_data.fillna({
            "guests" : 1,
            "beds" : 1,
            "bathrooms" : 1,
            "bedrooms" : 1,
        })
    
    def clean_tabular_data(self):
        """Groups all cleaning functions, and visualises the dataframe to ensure
        correct result of cleaning.
        """
        self.initial_cleaning()
        self.remove_rows_with_missing_ratings()
        self.clean_description_strings()
        self.set_default_feature_values()
        # Visualise DataFrame
        msno.matrix(self.listing_data)

    def save_df_as_csv(self):
        """Saves self.listing_data as csv file in current directory.
        """
        self.listing_data.to_csv("clean_listing.csv")
#%%
if __name__ == "__main__":
    # Initiate class and import raw data
    air_bnb_data = AirBnbDataPreparation()
    # clean raw data
    air_bnb_data.clean_tabular_data()
    # save clean DataFrame
    air_bnb_data.save_df_as_csv()