import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from tensorflow.keras.models import Sequential
from model_training import Model2BTrainer  

class TestModel2BTrainer(unittest.TestCase):

    def setUp(self):
        # Create mock datasets
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.eye(4)[np.random.choice(4, 100)]  # One-hot encoded labels
        self.X_val = np.random.rand(20, 10)
        self.y_val = np.eye(4)[np.random.choice(4, 20)]  # One-hot encoded labels
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.eye(4)[np.random.choice(4, 20)]  # One-hot encoded labels

        self.trainer = Model2BTrainer(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, learning_rate=0.001)

    def test_compute_class_weights(self):
        class_weights = self.trainer.compute_class_weights()
        self.assertIsInstance(class_weights, dict)
        self.assertEqual(len(class_weights), 4)  # Should be 4 classes

    def test_build_model(self):
        self.trainer.build_model()
        self.assertIsInstance(self.trainer.model, Sequential)
        self.assertEqual(len(self.trainer.model.layers), 5)  # Check the number of layers

    @patch.object(Model2BTrainer, 'build_model')
    def test_train_model(self, mock_build_model):
        mock_model = MagicMock()
        mock_build_model.return_value = mock_model
        
        # Mock the fit method
        mock_model.fit.return_value = MagicMock()
        
        self.trainer.train_model(epochs=5, batch_size=10)
        
        mock_build_model.assert_called_once()
        mock_model.fit.assert_called_once_with(
            self.X_train, self.y_train,
            epochs=5,
            batch_size=10,
            validation_data=(self.X_val, self.y_val),
            class_weight=self.trainer.class_weights,
            callbacks=unittest.mock.ANY
        )

    def test_evaluate_model(self):
        # Mock the model's history
        self.trainer.history = MagicMock()
        self.trainer.history.history = {'val_accuracy': [0.1, 0.2, 0.3, 0.4, 0.5]}
        
        self.trainer.evaluate_model()
        self.assertEqual(np.max(self.trainer.history.history['val_accuracy']), 0.5)

    @patch('tensorflow.keras.models.Sequential.save')
    def test_save_model(self, mock_save):
        file_path = 'test_model.h5'
        self.trainer.build_model()
        self.trainer.save_model(file_path)
        mock_save.assert_called_once_with(file_path)

    @patch('matplotlib.pyplot.show')
    def test_plot_learning_curves(self, mock_show):
        # Mock the model's history
        self.trainer.history = MagicMock()
        self.trainer.history.history = {
            'accuracy': [0.1, 0.2, 0.3, 0.4, 0.5],
            'val_accuracy': [0.15, 0.25, 0.35, 0.45, 0.55],
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3]
        }
        
        with patch('matplotlib.pyplot.plot') as mock_plot:
            self.trainer.plot_learning_curves()
            self.assertEqual(mock_plot.call_count, 8)  # There are 4 plots, each called twice (accuracy and loss)
            mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, mock_show):
        # Mock the model's predict method
        self.trainer.model = MagicMock()
        self.trainer.model.predict.return_value = np.eye(4)[np.random.choice(4, 20)]
        
        with patch('matplotlib.pyplot.gca') as mock_gca:
            self.trainer.plot_confusion_matrix()
            mock_gca.assert_called_once()
            mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
