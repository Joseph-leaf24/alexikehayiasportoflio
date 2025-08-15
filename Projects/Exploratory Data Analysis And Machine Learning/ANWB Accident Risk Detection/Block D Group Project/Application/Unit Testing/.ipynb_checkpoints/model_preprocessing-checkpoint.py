import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Model2BPreprocessor:
    def __init__(self, df, features, target='incident_severity'):
        """
        Initialize the Model2BPreprocessor class with the given dataset and parameters.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed.
            features (list): List of feature column names to be used for model training.
            target (str): The target column name to be mapped and encoded. Default is 'incident_severity'.
        """
        self.df = df
        self.features = features
        self.target = target
        self.scaler = None
        self.smote = SMOTE(random_state=42, k_neighbors=1)

    def map_severity(self):
        """
        Map the incident severity to reduced categories and encode them.
        """
        severity_mapping = {
            'HA1': 'low', 'HA2': 'low', 'HA3': 'low',
            'HB1': 'low', 'HB2': 'low', 'HB3': 'low',
            'HC1': 'low', 'HC2': 'low', 'HC3': 'low', 'HC4': 'low', 'HC5': 'low', 'HC6': 'low',
            'HC7': 'minor', 'HC8': 'minor', 'HC9': 'minor', 'HC10': 'minor',
            'HC11': 'moderate', 'HC12': 'moderate', 'HC13': 'moderate', 'HC14': 'moderate',
            'HC15': 'moderate', 'HC16': 'moderate', 'HC17': 'moderate', 'HC18': 'moderate',
            'HC19': 'severe', 'HC20': 'severe', 'HC21': 'severe'
        }
        self.df['reduced_severity'] = self.df[self.target].map(severity_mapping)
        label_encoder = LabelEncoder()
        self.df['encoded_severity'] = label_encoder.fit_transform(self.df['reduced_severity'])

    def create_accident_count(self):
        """
        Create a new column 'accident_count' based on the count of 'eventid'.
        """
        self.df['accident_count'] = self.df.groupby('eventid')['eventid'].transform('count')

    def handle_missing_values(self):
        """
        Convert features to numeric and handle missing values by filling them with zero.
        """
        self.df[self.features] = self.df[self.features].apply(pd.to_numeric, errors='coerce')
        self.df[self.features].fillna(0, inplace=True)

    def standardize_features(self):
        """
        Standardize the feature columns and return the scaler for future use.
        """
        self.scaler = StandardScaler()
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

    def balance_dataset(self):
        """
        Balance the dataset using SMOTE to handle class imbalance.
        """
        X_balanced, y_balanced = self.smote.fit_resample(self.df[self.features], self.df['encoded_severity'])

        # Filter out class 4 (if needed)
        filtered_indices = y_balanced < 4
        self.X_balanced = X_balanced[filtered_indices]
        self.y_balanced = y_balanced[filtered_indices]

    def split_dataset(self, test_size=0.1, val_size=0.2):
        """
        Split the dataset into training, validation, and test sets.

        Args:
            test_size (float): Fraction of the dataset to be used as the test set. Default is 0.1 (10%).
            val_size (float): Fraction of the remaining dataset to be used as the validation set. Default is 0.2 (20%).

        Returns:
            Tuple: The training, validation, and test sets.
        """
        # Split into temporary and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(self.X_balanced, self.y_balanced, test_size=test_size, random_state=42)

        # Split temporary set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

        # Convert labels to categorical
        y_train_encoded = to_categorical(y_train, num_classes=4)
        y_val_encoded = to_categorical(y_val, num_classes=4)
        y_test_encoded = to_categorical(y_test, num_classes=4)

        return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded

    def preprocess(self):
        """
        Execute all preprocessing steps in sequence.

        Returns:
            Tuple: The preprocessed training, validation, and test sets.
        """
        self.map_severity()
        self.create_accident_count()
        self.handle_missing_values()
        self.standardize_features()
        self.balance_dataset()
        return self.split_dataset()
