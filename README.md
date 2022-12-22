# modelling_air_bnb_property_listing_dataset
Data Science Specialization Project of AiCore Curriculum

## Milestone 1: Data Preparation
Technologies / Skills:
- Exploratory Data Analysis
    - Descriptive statistics (measures of central tendency and measures of dispersion)
    - Data types (Categorical, ordinal, nominal)
- Data Visualisation
    - Plotly, plotly.express (scatter plots, bar graphs, box plots, histograms, choropleths, Sankey diagrams)
    - Matplotlib
- Cleaning Data
    - Pandas DataFrame concepts and operations
    - Dealing with missing data (missingno, imputation, interpolation)
- Amazon Web Services (AWS)
    - AWS CLI (Command Line Interface)
    - Amazon RDS (Relational Database Service)
    - Amazon S3 (Simple Storage Service)
    - Amazon EC2 (Elastic Compute Cloud)
    - boto3 SDK for Python
- Image Data Manipulation
    - JPEG, PNG
    - opencv-python
    - PILLOW


The python files [prepare_tabular_data.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/prepare_tabular_data.py) and [prepare_image_data.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/prepare_image_data.py) deal with data preparation for their respective data types.

Simple Pandas DataFrame manipulations were used to deal with missing data, clean text data, and save the cleaned tabular data to a new csv file, named [clean_tabular_data.csv](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/tabular_data/clean_tabular_data.csv).

Images relevant to the AirBnB listings were saved on the cloud using Amazon Web Services S3 service. The ImageProcessing class in the [prepare_image_data.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/prepare_image_data.py) file contains methods using the boto3 Softward Development Kit packageto download the images, and rescale the images to the height of the smallest inage in the data set.

(screenshot of rescaled image?)

## Milestone 2: Creating a Regression Model
Technologies / Skills:







