import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class DeepLearningModel:
    def __init__(self, input_dim, output_dim=4, learning_rate=0.0003):
        """
        Initialize the deep learning model class.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes. Default is 4.
            learning_rate (float): Learning rate for the optimizer. Default is 0.0003.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """
        Build and compile the Keras Sequential model.

        Returns:
            Sequential: Compiled Keras Sequential model.
        """
        model = Sequential([
            Dense(1024, input_dim=self.input_dim, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.4),

            Dense(512, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.4),

            Dense(256, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.4),

            Dense(128, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.4),

            Dense(64, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.4),

            Dense(32, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.4),

            Dense(self.output_dim, activation='softmax')  # Output layer for the specified number of categories
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, validation_split=0.2, epochs=50, batch_size=256):
        """
        Train the model with the provided training data.

        Args:
            X_train (np.array): Training feature data.
            y_train (np.array): Training label data in categorical format.
            validation_split (float): Fraction of data to be used as validation data. Default is 0.2.
            epochs (int): Number of training epochs. Default is 50.
            batch_size (int): Number of samples per gradient update. Default is 256.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )

    def evaluate_model(self, X, y, dataset_type='Test'):
        """
        Evaluate the model on the given dataset.

        Args:
            X (np.array): Feature data.
            y (np.array): Label data in categorical format.
            dataset_type (str): Type of the dataset (e.g., 'Training' or 'Test'). Default is 'Test'.

        Returns:
            tuple: Loss and accuracy of the model on the given dataset.
        """
        loss, accuracy = self.model.evaluate(X, y)
        print(f'{dataset_type} Loss: {loss:.4f}')
        print(f'{dataset_type} Accuracy: {accuracy:.4f}')
        return loss, accuracy

    def predict_and_evaluate(self, X_test, y_test):
        """
        Generate predictions on the test set and evaluate the results.

        Args:
            X_test (np.array): Test feature data.
            y_test (np.array): Test label data in categorical format.
        """
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'Minor', 'Moderate', 'Severe'],
                    yticklabels=['Low', 'Minor', 'Moderate', 'Severe'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Classification report
        print(classification_report(y_true_classes, y_pred_classes,
                                    target_names=['Low', 'Minor', 'Moderate', 'Severe']))

    def plot_learning_curves(self):
        """
        Plot the learning curves for accuracy and loss over epochs.
        """
        plt.figure(figsize=(12, 6))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_model(self, file_path):
        """
        Save the trained model to a file.

        Args:
            file_path (str): Path to save the model file.
        """
        self.model.save(file_path)
        print(f'Model saved to {file_path}')