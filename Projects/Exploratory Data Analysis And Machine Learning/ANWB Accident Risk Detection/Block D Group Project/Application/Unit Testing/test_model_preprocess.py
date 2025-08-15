import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from model_preprocessing import Model2BPreprocessor  

class TestModel2BPreprocessor(unittest.TestCase):

    def setUp(self):
        # Create a mock DataFrame with relevant columns
        data = {
            'incident_severity': ['HA1', 'HC7', 'HC11', 'HC19', 'HA1', 'HC21'],
            'eventid': [1, 1, 2, 3, 3, 3],
            'latitude': [50.0, 51.0, np.nan, 49.5, 48.5, 47.0],
            'longitude': [5.0, 5.5, 6.0, np.nan, 4.0, 3.5],
            'speed_kmh': [70, 80, 60, 50, 75, np.nan],
            'maxwaarde': [200, 220, np.nan, 210, 230, 240]
        }
        self.df = pd.DataFrame(data)
        self.features = ['latitude', 'longitude', 'speed_kmh', 'maxwaarde', 'accident_count']
        self.preprocessor = Model2BPreprocessor(self.df, self.features)

    def test_map_severity(self):
        self.preprocessor.map_severity()
        expected_reduced_severity = ['low', 'minor', 'moderate', 'severe', 'low', 'severe']
        expected_encoded_severity = [0, 1, 2, 3, 0, 3]  # Assume encoder assigns 0,1,2,3 to low, minor, moderate, severe

        self.assertListEqual(self.preprocessor.df['reduced_severity'].tolist(), expected_reduced_severity)
        self.assertListEqual(self.preprocessor.df['encoded_severity'].tolist(), expected_encoded_severity)

    def test_create_accident_count(self):
        self.preprocessor.create_accident_count()
        expected_accident_count = [2, 2, 1, 3, 3, 3]
        self.assertListEqual(self.preprocessor.df['accident_count'].tolist(), expected_accident_count)

    def test_handle_missing_values(self):
        self.preprocessor.handle_missing_values()
        # Check if NaNs are replaced with zeros
        self.assertFalse(self.preprocessor.df[self.features].isnull().values.any())

    def test_standardize_features(self):
        self.preprocessor.standardize_features()
        # Ensure the scaler is fitted and the features are standardized
        self.assertIsInstance(self.preprocessor.scaler, StandardScaler)
        self.assertTrue(np.allclose(self.preprocessor.df[self.features].mean(), 0, atol=1e-1))
        self.assertTrue(np.allclose(self.preprocessor.df[self.features].std(), 1, atol=1e-1))

    def test_balance_dataset(self):
        self.preprocessor.map_severity()
        self.preprocessor.create_accident_count()
        self.preprocessor.handle_missing_values()
        self.preprocessor.standardize_features()
        self.preprocessor.balance_dataset()
        # Ensure that SMOTE resampling works correctly
        unique, counts = np.unique(self.preprocessor.y_balanced, return_counts=True)
        balanced_class_counts = dict(zip(unique, counts))
        self.assertTrue(all(count == balanced_class_counts[0] for count in balanced_class_counts.values()))

    def test_split_dataset(self):
        self.preprocessor.map_severity()
        self.preprocessor.create_accident_count()
        self.preprocessor.handle_missing_values()
        self.preprocessor.standardize_features()
        self.preprocessor.balance_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_dataset()

        # Check that the shapes match expected sizes
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))

        # Check if labels are one-hot encoded
        self.assertEqual(y_train.shape[1], 4)
        self.assertEqual(y_val.shape[1], 4)
        self.assertEqual(y_test.shape[1], 4)

    def test_preprocess(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.preprocess()

        # Check if the whole pipeline runs and outputs are as expected
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(y_train.shape[1], 4)
        self.assertEqual(y_val.shape[1], 4)
        self.assertEqual(y_test.shape[1], 4)

if __name__ == '__main__':
    unittest.main()
