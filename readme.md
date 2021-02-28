# Quality Control Preprocessing for High-Content Screening Experiments
The identification of systematic errors and random artifacts in microscopy images from high-content screening (HCS) experiments is critical for the acquisition of robust datasets. Common image-based aberrations may include out-of-focus images, debris, and autofluorescing samples which can cause artifacts such as excessive focus blur and image saturation. Manual inspection of images within a HCS experiment is simply not feasible but poor quality images must be appropriately identified and addressed as not to degrade otherwise high-quality data.

This quality control (QC) protocol is designed to calculate image-based measurements of QC metrics and apply a trained machine-learning model to identify, with high-confidence, images which may fail quality control assessment.

The example training dataset provided is generated using a QC pipeline built in CellProfiler, using 177 diverse images collected over a 2-year period including poor- and good-quality images. The images are raw grayscale TIF images of patient and patient-derived leukaemic cells, in coculture with mesenchymal stromal cells, assayed across a wide range of experimental conditions. Image metrics are then used to train an machine learning Voting Classifier model (an ensemble of classifiers).

> ***Important*** To apply this protocol to other datasets, it is important to generate unique training data as the model here would be
appropriate only to data acquired under the exact same experimental conditions.

## Contents
* ```Requirements``` contains the required dependencies used throughout the workflow.
* [QualityControl.cppipe](QualityControl.cppipe) is a quality control pipeline for CellProfiler used to calculate image quality metrics for analysis.
* [QC_predict.py](QC_predict.py) is the main script which trains the model from a training dataset and predicts upon unseen test data, outputting a list of image names which fail QC and require attention.
* [QC_demo.ipynb](QC_demo.ipynb) shows, step-by-step, the workflow of: importing the training data, training the model, assessing the performance of the model by plotting a confusion matrix and calculating performance metrics, and how to apply the trained model to unseen datasets.
* [QC_training.csv](QC_training.csv) is a demonstration Comma-Separated Values (CSV) file containing training data used to build the model. In practice, you must generate your own training data unique to your own experimental and/or imaging acquisition conditions.
* [raw_data/raw_data_example.csv](raw_data/raw_data_example.csv) is a CSV file to demonstrate reading and predicting on unseen data output from the ```QualityControl.cppipe``` pipeline.
## Getting Started
Walkthrough instructions are provided in the [QC_demo.ipynb](QC_demo.ipynb) Jupyter notebook which details how to:
1. generate the training set for building the machine learning model;
2. import the training data and configure it for training use
3. train the model *(hyper-parameters should be manually configured using GridSearch cross-validation testing to find optimal values)*
4. assess the performance of the model
5. predict QC on unseen datasets

Once optimal hyperparameters have been configured and a robust training dataset has been composed, ```QC_predict.py``` can be changed to reflect optimal parameters and applied to whole datasets.

## Dependencies
Image quality metrics are calculated using a pipeline built using [CellProfiler](https://github.com/CellProfiler) (4.0.5)
The script relies on the following dependencies (tested version provided in parentheses):
* python (3.8)
* pandas (1.1.4)
* scikit-learn (0.24.1)
* matplotlib (3.3.3)
* seaborn (0.11.0)

## Install
Clone this repository: git clone https://github.com/hayden-bell/HCS-QC