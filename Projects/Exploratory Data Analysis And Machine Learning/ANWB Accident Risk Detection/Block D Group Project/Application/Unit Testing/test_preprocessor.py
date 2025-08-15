import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor

print("Running test_preprocessor.py")

class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up TestDataPreprocessor")
        cls.db_params = {
            'host': 'test_host',
            'port': 'test_port',
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }
        cls.default_timestamp = pd.Timestamp('1970-01-01', tz='UTC')
        cls.landmark_coords = (51.5890, 4.7745)
        cls.columns_info = {
            "safe_driving": [
                ('eventid', 'INTEGER'),
                ('event_start', 'TIMESTAMP'),
                ('event_end', 'TIMESTAMP'),
                ('duration_seconds', 'REAL'),
                ('latitude', 'REAL'),
                ('longitude', 'REAL'),
                ('speed_kmh', 'REAL'),
                ('end_speed_kmh', 'REAL'),
                ('maxwaarde', 'REAL'),
                ('category', 'VARCHAR'),
                ('incident_severity', 'VARCHAR'),
                ('is_valid', 'BOOLEAN'),
                ('road_segment_id', 'INTEGER'),
                ('road_manager_type', 'VARCHAR'),
                ('road_number', 'VARCHAR'),
                ('road_name', 'VARCHAR'),
                ('place_name', 'VARCHAR'),
                ('municipality_name', 'VARCHAR'),
                ('road_manager_name', 'VARCHAR')
            ]
        }
        cls.preprocessor = DataPreprocessor(cls.db_params, cls.default_timestamp, cls.landmark_coords, cls.columns_info)

    def test_fetch_and_preprocess_data(self):
        print("Running test_fetch_and_preprocess_data")
        with patch.object(self.preprocessor, 'read_table', return_value=pd.DataFrame()):
            with patch.object(self.preprocessor, 'clean_dataframe', side_effect=lambda df, name: df):
                with patch.object(self.preprocessor, 'preprocess_dataframe', side_effect=lambda df, name: df):
                    safe_driving_df, breda_road_df, precipitation_df, temperature_df, greenery_df = self.preprocessor.fetch_and_preprocess_data()
                    
                    self.assertIsInstance(safe_driving_df, pd.DataFrame)
                    self.assertIsInstance(breda_road_df, pd.DataFrame)
                    self.assertIsInstance(precipitation_df, pd.DataFrame)
                    self.assertIsInstance(temperature_df, pd.DataFrame)
                    self.assertIsInstance(greenery_df, pd.DataFrame)

    @patch('data_preprocessing.psycopg2.connect')
    def test_connect_to_db(self, mock_connect):
        print("Running test_connect_to_db")
        # Mock successful database connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        conn = self.preprocessor.connect_to_db()
        self.assertEqual(conn, mock_conn)
        mock_connect.assert_called_once_with(**self.db_params)

        # Mock failed database connection
        mock_connect.side_effect = Exception("Connection failed")
        conn = self.preprocessor.connect_to_db()
        self.assertIsNone(conn)

    @patch('data_preprocessing.psycopg2.connect')
    def test_read_table(self, mock_connect):
        print("Running test_read_table")
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_df = pd.DataFrame({'eventid': [1, 2], 'event_start': [self.default_timestamp, self.default_timestamp]})
        with patch('pandas.read_sql_query', return_value=mock_df):
            query = "SELECT * FROM safe_driving"
            df = self.preprocessor.read_table(query)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(df.empty)

        # Test case where connection fails
        mock_connect.side_effect = Exception("Connection failed")
        df = self.preprocessor.read_table(query)
        self.assertTrue(df.empty)

    def test_clean_dataframe(self):
        print("Running test_clean_dataframe")
        raw_data = {
            'eventid': [1, 2, np.nan],
            'event_start': [None, '2024-06-13 12:00:00', 'invalid date'],
            'latitude': [51.6, 51.7, 51.8],
            'longitude': [4.7, 4.8, 4.9],
            'speed_kmh': [80, np.nan, 100],
            'category': [None, 'A', 'B']
        }
        df = pd.DataFrame(raw_data)
        cleaned_df = self.preprocessor.clean_dataframe(df, 'safe_driving')
    
        self.assertEqual(cleaned_df['eventid'].isnull().sum(), 0)
        self.assertEqual(cleaned_df['event_start'].isnull().sum(), 0)
        self.assertEqual(cleaned_df['latitude'].isnull().sum(), 0)
        self.assertEqual(cleaned_df['longitude'].isnull().sum(), 0)
        self.assertEqual(cleaned_df['speed_kmh'].isnull().sum(), 0)
        self.assertEqual(cleaned_df['category'].isnull().sum(), 0)
        self.assertIn('distance_to_landmark', cleaned_df.columns)  # Check if the column exists
        self.assertIn('time_of_day', cleaned_df.columns)
    

    def test_normalize_dataframe(self):
        print("Running test_normalize_dataframe")
        raw_data = {
            'latitude': [51.5, 51.6, 51.7],
            'longitude': [4.7, 4.8, 4.9],
            'speed_kmh': [50, 60, 70],
            'duration_seconds': [30, 45, 60]
        }
        df = pd.DataFrame(raw_data)

        normalized_df = self.preprocessor.normalize_dataframe(df, 'minmax')
        self.assertAlmostEqual(normalized_df['latitude'].max(), 1.0)
        self.assertAlmostEqual(normalized_df['longitude'].max(), 1.0)
        self.assertAlmostEqual(normalized_df['speed_kmh'].max(), 1.0)
        self.assertAlmostEqual(normalized_df['duration_seconds'].max(), 1.0)

        normalized_df = self.preprocessor.normalize_dataframe(df, 'standard')
        self.assertAlmostEqual(normalized_df['latitude'].mean(), 0.0, places=5)
        self.assertAlmostEqual(normalized_df['longitude'].mean(), 0.0, places=5)
        self.assertAlmostEqual(normalized_df['speed_kmh'].mean(), 0.0, places=5)
        self.assertAlmostEqual(normalized_df['duration_seconds'].mean(), 0.0, places=5)

        normalized_df = self.preprocessor.normalize_dataframe(df, 'robust')
        self.assertAlmostEqual(normalized_df['latitude'].median(), 0.0, places=5)
        self.assertAlmostEqual(normalized_df['longitude'].median(), 0.0, places=5)
        self.assertAlmostEqual(normalized_df['speed_kmh'].median(), 0.0, places=5)
        self.assertAlmostEqual(normalized_df['duration_seconds'].median(), 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
