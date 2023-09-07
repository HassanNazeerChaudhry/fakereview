import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from scipy import stats

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.X = None  # Features
        self.y = None  # Target variable
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.smote = SMOTE()
        self.outlier_threshold = 3

    def check_balance(self):
        # Check if the dataset is balanced
        class_counts = self.y.value_counts()
        return class_counts.min() / class_counts.max() > 0.8  # Adjust the threshold as needed

    def balance_data(self):
        # Apply SMOTE to balance the dataset
        self.X, self.y = self.smote.fit_resample(self.X, self.y)

    def scale_features(self):
        # Standardize features
        self.X = self.scaler.fit_transform(self.X)

    def fill_missing_data(self):
        # Fill missing data with the mean (you can adjust the strategy)
        self.X = self.imputer.fit_transform(self.X)

    def remove_outliers(self):
        # Remove outliers using Z-score method (you can adjust the threshold)
        z_scores = np.abs(stats.zscore(self.X))
        self.X = self.X[(z_scores < self.outlier_threshold).all(axis=1)]
        self.y = self.y[(z_scores < self.outlier_threshold).all(axis=1)]

    def split_data(self, test_size=0.2, random_state=None):
        # Split data into train and test sets
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def k_fold_cross_validation(self, model, k=5):
        # Perform k-fold cross-validation
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='accuracy')
        return scores.mean()
