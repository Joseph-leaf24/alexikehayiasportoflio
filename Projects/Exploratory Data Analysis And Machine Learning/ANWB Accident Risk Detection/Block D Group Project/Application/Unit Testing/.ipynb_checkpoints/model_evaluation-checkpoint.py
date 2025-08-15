import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

class Model2BEvaluator:
    def __init__(self, model, history, X_train, y_train, X_test, y_test):
        """
        Initialize the Model2BEvaluator class with the trained model and datasets.

        Args:
            model (Sequential): The trained Keras model.
            history (History): The history object returned by model.fit(), containing training metrics over epochs.
            X_train (np.array): Training features.
            y_train (np.array): Training labels (one-hot encoded).
            X_test (np.array): Test features.
            y_test (np.array): Test labels (non-one-hot encoded).
        """
        self.model = model
        self.history = history
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        """
        Evaluate the model on the training and test sets and print the accuracy.
        """
        self.train_loss, self.train_accuracy = self.model.evaluate(self.X_train, self.y_train)
        self.test_loss, self.test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Training Loss: {self.train_loss:.4f}, Training Accuracy: {self.train_accuracy:.4f}")
        print(f"Test Loss: {self.test_loss:.4f}, Test Accuracy: {self.test_accuracy:.4f}")

    def plot_learning_curves(self):
        """
        Plot the learning curves for training and test accuracy over epochs.
        """
        plt.figure(figsize=(12, 6))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy during Training')
        plt.plot([self.train_accuracy] * len(self.history.history['accuracy']), label='Final Train Accuracy', linestyle='--')
        plt.plot([self.test_accuracy] * len(self.history.history['accuracy']), label='Test Accuracy', linestyle='--')
        plt.title('Model2_B Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss during Training')
        plt.plot(self.history.history['val_loss'], label='Validation Loss during Training')
        plt.plot([self.train_loss] * len(self.history.history['loss']), label='Final Train Loss', linestyle='--')
        plt.plot([self.test_loss] * len(self.history.history['loss']), label='Test Loss', linestyle='--')
        plt.title('Model2_B Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        plt.show()

    def plot_confusion_matrix(self):
        """
        Predict on the test set and plot the confusion matrix.
        """
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)

        # Plot confusion matrix
        plt.figure(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues', ax=plt.gca())
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

        print("Confusion Matrix:")
        print(cm)

    def print_classification_report(self):
        """
        Print the classification report based on the test set predictions.
        """
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

        # Generate classification report
        class_report = classification_report(self.y_test, y_pred_classes, target_names=['Minor accident risk', 'Moderate accident risk', 'Severe accident risk', 'Extreme danger'])

        # Print the classification report
        print("Classification Report:")
        print(class_report)

    def run_evaluation(self):
        """
        Run all evaluation steps: evaluate model, plot learning curves, plot confusion matrix, and print classification report.
        """
        self.evaluate_model()
        self.plot_learning_curves()
        self.plot_confusion_matrix()
        self.print_classification_report()