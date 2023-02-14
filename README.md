# Modelling the Airbnb Property Listing Algorithm
Data Science Specialization Project of AiCore Curriculum. A dataset of information about Airbnb properties is prepared using data cleaning, visualisation, and exploratory analysis. The data is stored on the cloud using AWS S3, and then a suite of machine learning and deep learning models are built (both regression and classification) to explore possibilities for property listing and ranking. 

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

Images relevant to the AirBnB listings were saved on the cloud using Amazon Web Services S3 service. The ImageProcessing class in the [prepare_image_data.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/prepare_image_data.py) file contains methods using the boto3 Softward Development Kit packageto download the images, and rescale the images to the height of the smallest image in the data set.

The following is a plot of of all the features containing numerical data which were used for regression and classification modelling. Each feature is visualised seperately in a scatter plot against the target label, Price per Night:

![feature_visualisation](/project_files/utils/documentation_images/feature_visualisation.png?raw=True)

## Milestone 2: Regression and Classification Models
Technologies / Skills:
 - Regression and Classification Models:
    - Linear regression
    - Logistic regression
    - Decision trees
    - Random forests with bagging
    - Adaboost
    - Gradient boost
- Associated skills:
    - Gradient descent 
    - Validation and testing
    - Bias vs variance tradeoff
    - Hyperparameter optimisation 
    - Cross validation
    - Regularisation tecniques
    - Maximum Likelihood estimation
- Technologies:
    - scikit-learn
    - numba

### Regression

Various regression models were trained in this section on the data prepared in the last in an attempt to accurately predict the nightly cost of a property based on the features in the dataset.

Four main models were tested: stochastic gradient descent, simple decision tree, random forest, and gradient boost. These were implemented using the module scikit-learn as shown in the files [modelling.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/modelling.py) and [find_best_regression_model.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/find_best_regression_model.py).

The accuracy scores of these models are as follows:

| Model          |      R^2 Score     | 
|----------------|--------------------|
| SGDRegression  |       0.343        | 
| Decision Tree  |       0.216        | 
| Random Forest  |       0.355        | 
| Gradient Boost |       0.362        |

Clearly these tecniques are insufficient for robust predictions of the target labels.

### Classification

Classification versions of decision tree, random forest, and gradient boost models were trained to predict the property category in an example of multiclass classification.

Overfitting was reduced by optimising hyperparameters such as number of estimators, number of features, and maximum tree depth.

The best model was selected based on the F1 score on predicitons of the validation set, and are as follows:

|     Model      |   F1 Accuracy Score   | 
|----------------|-----------------------|
| Decision Tree  |         0.328         | 
| Random Forest  |         0.381         | 
| Gradient Boost |         0.339         |

Again, however, these tecniques are insufficient for robust classifications of the target labels.

## Milestone 3: Deep Learning Models with PyTorch
Technologies / Skills:
 - PyTorch and Neural Networks
    - Tensors
    - Datasets and DataLoaders
    - Transforms
    - Activation functions
    - Backpropagation
    - Tensorboard

### Regression

A simple neural network was implemented in PyTorch with the aim of improving on the machine learning regression model predictions for nightly property price. The code implementing the dataset and neural network classes, as well as the training and evaluation functions can be found in the [deep_learning_regression_models.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/deep_learning_regression_models.py) file. Various numbers of hidden layers, number of hidden layer nodes, activation functions, and optimisers were tested, as well as extensive testing of learning rates and number of epochs. The performance metrics of the best model are as follows, along with a visualisation of the training loss in tensorboard.

| Model Parameters   |      Value         | 
|--------------------|--------------------|
| # Hidden Layers    |       2            | 
| Hidden Layer Width |       128          | 
| Optimiser          |       Adam         | 
| Epochs             |       26           |
| Learning Rate      |       0.001        |
| R^2 Score          |       0.372        |

![best_price_nights_model.png](/project_files/utils/documentation_images/best_price_nights_model.png?raw=True)

### Classification

The pipeline was reused for a classification problem, this time predicting the number of bedrooms in a property. Hence minor changes were required, such as the use of the Cross Entropy loss function and the F1 score metric for model evaluation. The code for this multiclass classification analysis can be found here: [deep_learning_classification_models.py](https://github.com/tuttonluke/modelling_air_bnb_property_listing_dataset/blob/main/project_files/deep_learning_classification_models.py).

The performance metrics for the best model are as follows:

| Model Parameters   |      R^2 Score     | 
|--------------------|--------------------|
| # Hidden Layers    |       2            | 
| Hidden Layer Width |       128          | 
| Optimiser          |       Adam         | 
| Epochs             |       19           |
| Learning Rate      |       0.001        |
| F1 Score           |       0.389        |

The training and validation accuracy and losses were visualised, along with the confusion matrix of predictions on the test data set. The proficiency of the model is clearly lacking, but there is a small improvement over the machine learning models tested previously, as described above.

![loss_visualisaiton.png](/project_files/deep_learning_models/classification/2023-02-14_10.16.18.178333/loss_visualisation.png?raw=True)

![confusion_matrix.png](/project_files/deep_learning_models/classification/2023-02-14_10.16.18.178333/confusion_matrix.png?raw=True)