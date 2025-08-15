import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.utils import to_categorical

class Model2BTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, learning_rate=0.001):
        """
        Initialize the Model2BTrainer with the given dataset and parameters.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels (one-hot encoded).
            X_val (np.array): Validation features.
            y_val (np.array): Validation labels (one-hot encoded).
            X_test (np.array): Test features.
            y_test (np.array): Test labels (one-hot encoded).
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.class_weights = self.compute_class_weights()

    def compute_class_weights(self):
        """
        Compute class weights to handle class imbalance in the training data.
        """
        y_train_classes = np.argmax(self.y_train, axis=1)  # Convert one-hot encoded labels to class indices
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_classes), y=y_train_classes)
        class_weights_dict = dict(enumerate(class_weights))
        return class_weights_dict

    def build_model(self):
        """
        Build and compile the neural network model.
        """
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(self.X_train.shape[1],), kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(self.y_train.shape[1], activation='softmax'))
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train_model(self, epochs=60, batch_size=64):
        """
        Train the model with the provided training data.

        Args:
            epochs (int): Number of epochs for training. Default is 60.
            batch_size (int): Batch size for training. Default is 64.
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            class_weight=self.class_weights,
            callbacks=callbacks
        )

    def evaluate_model(self):
        """
        Evaluate the model on the validation data and print the best validation accuracy.
        """
        val_accuracy = np.max(self.history.history['val_accuracy'])
        print(f"Model validation accuracy: {val_accuracy}")

    def save_model(self, file_path):
        """
        Save the trained model to a file.

        Args:
            file_path (str): Path to save the model file.
        """
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def plot_learning_curves(self):
        """
        Plot the learning curves for training and validation accuracy and loss.
        """
        plt.figure(figsize=(12, 5))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the test set predictions.
        """
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
        y_test_classes = np.argmax(self.y_test, axis=1)  # Convert one-hot encoded labels to class indices

        cm = confusion_matrix(y_test_classes, y_pred_classes)

        plt.figure(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues', ax=plt.gca())
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

        print("Confusion Matrix:")
        print(cm)
