#%%
import pandas as pd
import missingno as msno
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
        # Remove empty column
        self.listing_data = self.listing_data.drop("Unnamed: 19", axis=1)

    def remove_rows_with_missing_ratings(self) -> pd.DataFrame:
        """Removes rows from the self.listing_data that have missing ratings data.
        """
        self.listing_data = self.listing_data.dropna(subset=[
                                                    "Cleanliness_rating", 
                                                    "Accuracy_rating",
                                                    "Communication_rating",
                                                    "Location_rating",
                                                    "Check-in_rating",
                                                    "Value_rating"
                                                    ])
        # Visualise DataFrame to ensure the correct result.
        msno.matrix(self.listing_data)
    
    def combine_description_strings(self):
        pass
#%%
if __name__ == "__main__":
    air_bnb_data = AirBnbDataPreparation()
    air_bnb_data.initial_cleaning()
    air_bnb_data.remove_rows_with_missing_ratings()

    df = air_bnb_data.listing_data
