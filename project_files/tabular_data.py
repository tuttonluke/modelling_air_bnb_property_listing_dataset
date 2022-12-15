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
        # Visualise DataFrame to ensure the correct result.
        msno.matrix(self.listing_data)
    
    def clean_description_strings(self):
        for index in range(len(self.listing_data["Description"])):
            self.listing_data["Description"][index] = literal_eval(self.listing_data["Description"][index])            

    def save_df_as_csv(self):
        """Saves self.listing_data as csv file in current directory.
        """
        self.listing_data.to_csv("new_listing_data.csv")

#%%
if __name__ == "__main__":
    air_bnb_data = AirBnbDataPreparation()
    air_bnb_data.initial_cleaning()
    air_bnb_data.remove_rows_with_missing_ratings()
    air_bnb_data.clean_description_strings()

    # air_bnb_data.save_df_as_csv()
    df = air_bnb_data.listing_data
#%%