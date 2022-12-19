#%%
import boto3
import pandas as pd
import os
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
        """Downloads images from s3 bucket, all images from a single
        property being saved into a folder named with the property ID.
        """
        for element in self.list_of_folders[:2]:
            os.mkdir(f"images/{element}")
            list_of_files = []
            response = self.client.list_objects_v2(Bucket=self.bucket_name,
                                                    Prefix=f"{element}")
            objects = response["Contents"]
            for object in objects:
                list_of_files.append(object["Key"])
            for file in list_of_files:
                self.client.download_file(
                    self.bucket_name,
                    f"{file}",
                    f"images/{file}"
                )

    def resize_images(self):
        pass
# %%
if __name__ == "__main__":
    image_processor = ImageProcessing()
    image_processor.get_list_of_folders()
    image_processor.download_images()                                  
