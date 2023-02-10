#%%
import boto3
import pandas as pd
import os
import cv2 as cv
#%%
class ImageProcessing:
    def __init__(self) -> None:
        self.client = boto3.client("s3")
        self.bucket_name = "tuttonluke-airbnb-images"
        self.list_of_folders = []
        self.image_df = pd.DataFrame()

    def get_list_of_folders(self):
        """Populates self.list_of_folders with all folder names in s3 drive.
        """
        response = self.client.list_objects_v2(Bucket=self.bucket_name,
                                                Delimiter="/")
        for prefix in response["CommonPrefixes"]:
            self.list_of_folders.append(prefix["Prefix"][:-1])

    def get_image_df_info(self):
        """Populates self.image_df with information about all images:
        file path, image height, image width.
        """
        list_of_file_paths = []
        list_of_heights = []
        list_of_widths = []
        # List of file paths, image height, and image width
        for element in self.list_of_folders:
            list_of_files = []
            folders = self.client.list_objects_v2(Bucket="tuttonluke-airbnb-images",
                                                Prefix=f"{element}"
                                                )
            objects = folders["Contents"]
            for object in objects:
                list_of_files.append(object["Key"])
            for file in list_of_files:
                list_of_file_paths.append(file)
                img = cv.imread(f"images/{file}")
                list_of_heights.append(img.shape[0])
                list_of_widths.append(img.shape[1])
        # add info to image_df
        self.image_df["File Path"]  = list_of_file_paths
        self.image_df["Height"]  = list_of_heights
        self.image_df["Width"]  = list_of_widths

    def download_images(self):
        """Downloads images from s3 bucket, all images from a single
        property being saved into a folder named with the property ID.
        """
        for element in self.list_of_folders:
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
        """Resises all images to same height and saves the processed image
        in processed_images folder. Resizing process maintains aspect ratio. All 
        images in RGB format.
        """
        min_height = self.image_df["Height"].min()
        for image_file_path in self.image_df["File Path"]:
            # read in image
            img = cv.imread(f"images/{image_file_path}")
            # Get image dimensions
            original_height = img.shape[0]
            original_width = img.shape[1]
            # rescale image
            scale_percent = min_height / original_height
            resized_height = min_height
            resized_width = int(original_width * scale_percent)
            resized_img = cv.resize(img, (resized_width, resized_height), 
                                    interpolation = cv.INTER_AREA)
            # save image
            save_file = image_file_path[37:]
            cv.imwrite(f"processed_images/{save_file}", 
                        resized_img)
# %%
if __name__ == "__main__":
    image_processor = ImageProcessing()
    image_processor.get_list_of_folders()
    image_processor.get_image_df_info()
    # image_processor.download_images()
    image_processor.resize_images()

    image_df = image_processor.image_df