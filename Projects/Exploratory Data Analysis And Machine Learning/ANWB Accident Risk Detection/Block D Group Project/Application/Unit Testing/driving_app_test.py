import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from driving_app import DrivingRiskApp  

class TestDrivingRiskApp(unittest.TestCase):
    def setUp(self):
        """Setup test environment for each test."""
        self.app = DrivingRiskApp()

    def test_set_bg_hack_url(self):
        """Test the set_bg_hack_url method."""
        with patch('streamlit.markdown') as mock_markdown:
            self.app.set_bg_hack_url()
            mock_markdown.assert_called_once()

    def test_load_pretrained_model(self):
        """Test the load_pretrained_model method."""
        with patch('tensorflow.keras.models.Sequential.load_weights') as mock_load_weights:
            self.app.load_pretrained_model()
            self.assertIsNotNone(self.app.model)
            mock_load_weights.assert_called_once_with('MLP_Best_Model_4.h5')
            self.assertEqual(self.app.model.optimizer.learning_rate.numpy(), 0.0001)

    def test_get_provided_dummy_data(self):
        """Test the get_provided_dummy_data method."""
        dummy_data = self.app.get_provided_dummy_data()
        self.assertIsInstance(dummy_data, pd.DataFrame)
        self.assertIn('latitude', dummy_data.columns)

    def test_predict_risk(self):
        """Test the predict_risk method."""
        # Mock the scaler and model for predict_risk
        self.app.scaler.transform = MagicMock(return_value=np.random.rand(1, 5))
        self.app.model = MagicMock()
        self.app.model.predict = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]]))
        
        data = pd.DataFrame({
            'latitude': [51.57965],
            'longitude': [4.803370],
            'speed_kmh': [63.997307],
            'maxwaarde': [71.150510],
            'accident_count': [0]
        })
        prediction, risk_level = self.app.predict_risk(data)

        self.app.scaler.transform.assert_called_once_with(data)
        self.app.model.predict.assert_called_once()
        self.assertEqual(risk_level, 'Severe accident risk')

    @patch('streamlit.text_input', return_value='user')
    @patch('streamlit.columns')
    @patch('streamlit.experimental_rerun')
    def test_login_page(self, mock_rerun, mock_columns, mock_text_input):
        """Test the login_page method."""
        self.app.login_page()
        self.assertTrue(mock_text_input.called)

    @patch('streamlit.button', return_value=True)
    @patch('streamlit.columns')
    @patch('streamlit.text_input', return_value='user')
    @patch('streamlit.experimental_rerun')
    def test_successful_login(self, mock_rerun, mock_text_input, mock_columns, mock_button):
        """Test successful login."""
        self.app.login_page()
        self.assertTrue(st.session_state['logged_in'])
        mock_rerun.assert_called_once()

    def test_home_page(self):
        """Test the home_page method."""
        with patch('streamlit.header') as mock_header:
            with patch('streamlit.write') as mock_write:
                with patch('streamlit.button') as mock_button:
                    mock_button.side_effect = [True, False]
                    self.app.home_page()
                    mock_header.assert_called_once()
                    mock_write.assert_called_once()
                    mock_button.assert_called()

    @patch('time.sleep', return_value=None)
    def test_analyzing_page(self, mock_sleep):
        """Test the analyzing_page method."""
        with patch('streamlit.spinner'):
            with patch('streamlit.experimental_rerun') as mock_rerun:
                self.app.analyzing_page()
                mock_sleep.assert_called_once_with(2)
                mock_rerun.assert_called_once()

    @patch('streamlit.success')
    @patch('streamlit.write')
    @patch('streamlit.columns')
    @patch('streamlit.empty')
    @patch('streamlit.experimental_rerun')
    def test_results_page(self, mock_rerun, mock_empty, mock_columns, mock_write, mock_success):
        """Test the results_page method."""
        self.app.model = MagicMock()
        self.app.scaler.fit = MagicMock()
        self.app.scaler.transform = MagicMock(return_value=np.random.rand(1, 5))
        self.app.model.predict = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.4]]))
        
        self.app.results_page()
        mock_success.assert_called_once_with("Predicted Risk Level: Severe accident risk")
        self.assertTrue(mock_write.called)

if __name__ == '__main__':
    unittest.main()
