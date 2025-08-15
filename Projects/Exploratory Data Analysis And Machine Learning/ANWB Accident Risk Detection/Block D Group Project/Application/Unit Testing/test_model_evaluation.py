import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from model_evaluation import Model2BEvaluator  

class TestModel2BEvaluator(unittest.TestCase):

    def setUp(self):
        # Create mock datasets
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.eye(4)[np.random.choice(4, 100)]  # One-hot encoded labels
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.choice(4, 20)  # Non-one-hot encoded labels

        # Create a mock model with a predict method
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.eye(4)[np.random.choice(4, 20)]
        self.mock_model.evaluate.return_value = (0.5, 0.8)  # Mocked loss and accuracy

        # Mock training history
        self.history = MagicMock()
        self.history.history = {
            'accuracy': [0.1, 0.2, 0.3, 0.4, 0.5],
            'val_accuracy': [0.15, 0.25, 0.35, 0.45, 0.55],
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3]
        }

        self.evaluator = Model2BEvaluator(self.mock_model, self.history, self.X_train, self.y_train, self.X_test, self.y_test)

    def test_evaluate_model(self):
        self.evaluator.evaluate_model()
        # Check if the evaluate method was called twice (for train and test datasets)
        self.assertEqual(self.mock_model.evaluate.call_count, 2)
        self.mock_model.evaluate.assert_any_call(self.X_train, self.y_train)
        self.mock_model.evaluate.assert_any_call(self.X_test, self.y_test)

    @patch('matplotlib.pyplot.show')
    def test_plot_learning_curves(self, mock_show):
        with patch('matplotlib.pyplot.plot') as mock_plot:
            self.evaluator.plot_learning_curves()
            # There are 8 calls to plot: 4 for accuracy and 4 for loss
            self.assertEqual(mock_plot.call_count, 8)
            mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, mock_show):
        with patch('matplotlib.pyplot.gca') as mock_gca:
            self.evaluator.plot_confusion_matrix()
            mock_gca.assert_called_once()
            mock_show.assert_called_once()
            # Check if the confusion matrix plot is created
            self.assertTrue(mock_show.called)

    @patch('builtins.print')
    def test_print_classification_report(self, mock_print):
        self.evaluator.print_classification_report()
        # Check if the print function is called and includes the expected strings
        self.assertTrue(any("Classification Report:" in args[0] for args in mock_print.call_args_list))
        self.assertTrue(any("precision" in args[0] for args in mock_print.call_args_list))

    def test_run_evaluation(self):
        with patch.object(Model2BEvaluator, 'evaluate_model', wraps=self.evaluator.evaluate_model) as mock_evaluate_model:
            with patch.object(Model2BEvaluator, 'plot_learning_curves', wraps=self.evaluator.plot_learning_curves) as mock_plot_learning_curves:
                with patch.object(Model2BEvaluator, 'plot_confusion_matrix', wraps=self.evaluator.plot_confusion_matrix) as mock_plot_confusion_matrix:
                    with patch.object(Model2BEvaluator, 'print_classification_report', wraps=self.evaluator.print_classification_report) as mock_print_classification_report:
                        self.evaluator.run_evaluation()
                        mock_evaluate_model.assert_called_once()
                        mock_plot_learning_curves.assert_called_once()
                        mock_plot_confusion_matrix.assert_called_once()
                        mock_print_classification_report.assert_called_once()

if __name__ == '__main__':
    unittest.main()
