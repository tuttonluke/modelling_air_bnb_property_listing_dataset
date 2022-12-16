#%%
import boto3
import pandas as pd
#%%
class ImageProcessing:
    def __init__(self) -> None:
        self.client = boto3.client("s3")
        self.bucket_name = "tuttonluke-airbnb-images"
        self.list_of_folders = []

    def get_list_of_folders(self):
        """Populates self.list_of_folders with all folder IDs in the
        s3 bucket.
        """
        response = self.client.list_objects_v2(Bucket=self.bucket_name,
                                                Delimiter="/")
        for prefix in response["CommonPrefixes"]:
            self.list_of_folders.append(prefix["Prefix"][:-1])

    def download_images(self):
        pass

    def resize_images(self):
        pass
# %%
if __name__ == "__main__":
    image_processor = ImageProcessing()
    image_processor.get_list_of_folders()

    print(image_processor.list_of_folders)                                      
#%%