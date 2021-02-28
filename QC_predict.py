import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier

# #############################################################################
# Training data

# Load training data generated using the QualityControl.cppipe CellProfiler pipeline
features = pd.read_csv('QC_training.csv')

# Labels are the values we are predicting
labels = np.array(features['Pass'])

# Remove the labels from the feature set
features = features.drop('Pass', axis=1)

feature_list = list(features.columns)
features = np.array(features)

# #############################################################################
# Training models

# Ensemble classification models. Hyper-parameters of each model should be tuned by cross-validation using GridSearchCV.
rf = RandomForestClassifier(n_estimators=350, min_samples_leaf=2, max_features='log2')
knn = KNN(n_neighbors=5)
dt = DecisionTreeClassifier(min_samples_leaf=0.2, max_depth=4, splitter='best')
ada = AdaBoostClassifier()
gbb = GradientBoostingClassifier()

classifiers = [('Random Forest', rf),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt),
               ('AdaBoost Classifier', ada),
               ('Gradient Boosting Classifier', gbb)]

# Fit each model to the training set
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(features, labels)

# Instantiate a VotingClassifier (vc)
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(features, labels)

# #############################################################################
# Predict on raw dataset

# Load raw data
raw_df = pd.read_csv('raw_data/raw_data_example.csv')

# Extract image metadata
image_no = np.array(raw_df['ImageNumber'])
image_name = np.array(raw_df['FileName_Image_QC'])

# Extract data 'features' for prediction to array
raw_df = raw_df.loc[:, 'ImageQuality_Correlation_Image_QC_16':'ImageQuality_TotalIntensity_Image_QC']
raw_features = np.array(raw_df)

# Predict on dataset
raw_pred = vc.predict(raw_features)

# Construct df to export predictions
raw_pred_df = pd.DataFrame()
raw_pred_df['ImageNumber'] = image_no
raw_pred_df['FileName'] = image_name
raw_pred_df['QC'] = raw_pred
raw_pred_df['QC'] = raw_pred_df['QC'].replace({1: 'Pass', 0: 'Fail'})
raw_pred_df.set_index('ImageNumber', inplace=True)

# Export failed images only
failed_df = raw_pred_df[raw_pred_df['QC'] == 'Fail']
if failed_df.shape[0] > 0:
    failed_df.to_csv('image_QC_failed.csv')
    if failed_df.shape[0] > 1:
        print(f'{failed_df.shape[0]} images failed QC checks in this dataset.')
    else:
        print(f'{failed_df.shape[0]} image failed QC checks in this dataset.')
else:
    print('There are no images that have not passed QC checks in this dataset.')
