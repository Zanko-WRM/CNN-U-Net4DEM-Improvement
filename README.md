# Improving Spatial Resolution and Accuracy of Global Digital Elevation Models for Flood Modeling Using Deep Learning-Based Integration of Remote Sensing Data 

# Overview
This GitHub repository contains the source code and datasets used in the research paper, "Improving Spatial Resolution and 
This GitHub repository contains the source code and datasets used in the research paper, "Improving Spatial Resolution and Accuracy of Global Digital Elevation Models for Flood Modeling Using Deep Learning-Based Integration of Remote Sensing Data." The paper introduces a deep learning-based approach to generate high-resolution Digital Elevation Models (DEMs) by leveraging multi-source data, including GDEMs (ALOS, ASTER, NASADEM, SRTM), remote imagery data (Landsat-8, Sentinel-1, Sentinel-2), and OpenStreetMap layers. This method utilizes a Convolutional Neural Network (CNN) U-Net model to integrate and refine these diverse data sources, producing enhanced DEMs at a 5-meter resolution. This approach significantly improves the accuracy and resolution of global DEMs, leading to more precise models essential for urban flood management and simulations.

# Key Contributions
- A high-resolution DEM was developed for urban flood modeling using deep learning-based multi-source data integration.
- The enhanced DEM significantly improved accuracy, showing reductions in RMSE and MAE, and detailed urban feature representation.
- The improved DEM demonstrated better flood pattern accuracy by increasing detection probability and reducing false alarms.

# Repository Structure
 - config.py: Contains configuration settings and constants used throughout the project, including default values for patches, batch sizes, and number of epochs. Note that hyperparameters for evaluation are also defined separately within the hyperparameter tuning script

- data_preparation.py: Includes functions and methods for reading, preprocessing, and splitting the data into training, validation, and test sets. It also handles patch creation and data normalization.

- visualization.py: Provides functions for visualizing various aspects of the data and model performance, including plotting training and validation metrics, DEMs, and evaluation metrics.

- cnn_unet.py: Contains the implementation of the Convolutional Neural Network (CNN) U-Net model used for improving the DEM resolution. It also includes loss functions and evaluation metrics.

- hyperparameter_tuning.py: Implements the hyperparameter tuning process using Optuna to optimize the U-Net model's performance by adjusting various hyperparameters.

- training.py: Script for training the U-Net model using the prepared datasets and optimized hyperparameters. It includes the training loop and validation steps.

- testing.py: Script for testing the trained U-Net model on new data. It includes methods for making predictions and evaluating the model's performance on the test set.

- evaluation.py: Contains functions for evaluating the enhanced DEM against reference DEMs and performing statistical error analysis and flood modeling performance assessment.
