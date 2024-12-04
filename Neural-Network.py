"""
Neural Network Training Script

This script provides functionality to train, evaluate, and make predictions with a neural network model.
It includes data preprocessing, model training with hyperparameter tuning, cross-validation, and prediction on new data.

Usage:
    python script.py --mode train --data_path data.csv

Arguments:
    --mode: Mode to run. Options are 'train', 'cross_validate', 'predict'.
    --data_path: Path to the dataset CSV file (required for 'train' and 'cross_validate' modes).
    --model_path: Path to the trained model file (required for 'predict' mode).
    --pipeline_path: Path to the preprocessing pipeline file (required for 'predict' mode).
    --sample_data: Comma-separated feature values for prediction (required for 'predict' mode).

Examples:
    Training mode:
        python script.py --mode train --data_path data.csv

    Cross-validation mode:
        python script.py --mode cross_validate --data_path data.csv

    Prediction mode:
        python script.py --mode predict --model_path model.h5 --pipeline_path pipeline.pkl --sample_data "0.5,1.2,3.4"

Author: Your Name
Date: YYYY-MM-DD
"""

import argparse
import datetime
import logging
import os
import random
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to set. Defaults to 42.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seed set to {seed}")


def check_gpu() -> None:
    """
    Checks if a GPU is available and logs the information.

    Logs:
        - GPU details if available.
        - A warning if no GPU is found.

    Returns:
        None
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU(s) available: {gpus}")
    else:
        logger.warning("No GPU available. Training will be performed on CPU.")


class DataPreprocessor:
    """
    Handles data preparation and preprocessing tasks.

    Attributes:
        df (pd.DataFrame): The input data frame.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        pipeline (Pipeline): The preprocessing pipeline.
    """

    def __init__(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the DataPreprocessor with data and configurations.

        Args:
            df (pd.DataFrame): The input data frame.
            target_column (str): The name of the target column.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None

    def prepare_data(self) -> tuple:
        """
        Prepares data for model training and evaluation.
        Splits data into training and testing sets and applies preprocessing.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Processed training features, test features, training labels, test labels.

        Raises:
            ValueError: If the target column is not in the DataFrame.
        """
        if self.target_column not in self.df.columns:
            logger.error(f"Target column '{self.target_column}' not found in DataFrame.")
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # Data validation: check for missing values
        if self.df.isnull().values.any():
            logger.warning("Missing values detected in the dataset. Imputing missing values.")
        else:
            logger.info("No missing values detected in the dataset.")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Create preprocessing pipeline
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])

        # Fit and transform training data
        X_train_processed = self.pipeline.fit_transform(X_train)

        # Only transform test data (do not fit)
        X_test_processed = self.pipeline.transform(X_test)

        # Save the pipeline
        self.save_pipeline()

        return X_train_processed, X_test_processed, y_train.values, y_test.values

    def save_pipeline(self) -> None:
        """
        Saves the preprocessing pipeline to a file.

        Returns:
            None
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            pipeline_dir = os.path.join('artifacts', 'pipelines')
            os.makedirs(pipeline_dir, exist_ok=True)
            pipeline_filename = os.path.join(pipeline_dir, f'pipeline_{timestamp}.pkl')
            joblib.dump(self.pipeline, pipeline_filename)
            logger.info(f"Pipeline saved as {pipeline_filename}")
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise


class NeuralNetworkModel:
    """
    Class for creating, training, and evaluating the neural network model.

    Attributes:
        input_dim (int): The number of features in the input data.
        class_weights (dict): Class weights to handle imbalanced datasets.
        model (tf.keras.Model): The Keras model.
        history: Training history.
        y_test (np.ndarray): True labels for test data.
        y_pred (np.ndarray): Predicted labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
    """

    def __init__(self, input_dim: int, class_weights: dict = None):
        """
        Initializes the NeuralNetworkModel with the input dimension and class weights.

        Args:
            input_dim (int): The number of features in the input data.
            class_weights (dict, optional): Class weights to handle imbalanced datasets. Defaults to None.
        """
        self.input_dim = input_dim
        self.class_weights = class_weights
        self.model = None
        self.history = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.best_hyperparameters = None

    def build_model(self, hp) -> tf.keras.Model:
        """
        Builds the Keras model with hyperparameters.

        Args:
            hp: Hyperparameter object from Keras Tuner.

        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        hp_units_input = hp.Int('units_input', min_value=32, max_value=128, step=32)
        hp_units_hidden = hp.Int('units_hidden', min_value=16, max_value=64, step=16)
        hp_dropout = hp.Float('dropout', 0.2, 0.5, step=0.1)
        hp_l2 = hp.Choice('l2', values=[0.0, 0.01, 0.1])
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

        model = Sequential()
        model.add(Dense(hp_units_input, activation='relu', input_dim=self.input_dim, kernel_regularizer=l2(hp_l2)))
        model.add(BatchNormalization())
        model.add(Dropout(hp_dropout))
        model.add(Dense(hp_units_hidden, activation='relu', kernel_regularizer=l2(hp_l2)))
        model.add(BatchNormalization())
        model.add(Dropout(hp_dropout))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Performs hyperparameter tuning using Keras Tuner.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.

        Returns:
            None
        """
        logger.info("Starting hyperparameter tuning...")

        try:
            import kerastuner as kt

            tuner = kt.RandomSearch(
                self.build_model,
                objective='val_accuracy',
                max_trials=5,
                executions_per_trial=1,
                directory='hyperparam_tuning',
                project_name='nn_tuning'
            )

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            tuner.search(
                X_train, y_train,
                epochs=50,
                validation_split=0.2,
                callbacks=[early_stopping],
                class_weight=self.class_weights,
                verbose=1
            )

            self.best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            logger.info(f"Best hyperparameters: {self.best_hyperparameters.values}")

            self.model = tuner.hypermodel.build(self.best_hyperparameters)

        except ImportError as e:
            logger.error(f"Keras Tuner is not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Trains the model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation features. Defaults to None.
            y_val (np.ndarray, optional): Validation labels. Defaults to None.

        Returns:
            None
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        callbacks = [early_stopping, reduce_lr]

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        try:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=validation_data,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluates the model on the test set.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            None
        """
        try:
            y_pred_proba = self.model.predict(X_test).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)

            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f'Test Loss: {loss:.4f}')
            logger.info(f'Test Accuracy: {accuracy:.4f}')

            report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            logger.info(f'\nClassification Report:\n{report}')
            logger.info(f'ROC AUC Score: {roc_auc:.4f}')

            self.y_test = y_test
            self.y_pred = y_pred
            self.y_pred_proba = y_pred_proba

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def save_model(self) -> None:
        """
        Saves the trained model to a file.

        Returns:
            None
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_dir = os.path.join('artifacts', 'models', timestamp)
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(model_dir, 'model.h5')
            self.model.save(model_filename)
            logger.info(f"Model saved as {model_filename}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def plot_training_history(self) -> None:
        """
        Plots training and validation loss and accuracy over epochs.

        Returns:
            None
        """
        try:
            history = self.history
            output_dir = os.path.join('artifacts', 'plots')
            os.makedirs(output_dir, exist_ok=True)

            # Plot loss over epochs
            plt.figure()
            plt.plot(history.history['loss'], label='Train Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss Over Epochs')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
            plt.savefig(loss_plot_path)
            plt.close()

            # Plot accuracy over epochs
            plt.figure()
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy Over Epochs')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            acc_plot_path = os.path.join(output_dir, 'accuracy_plot.png')
            plt.savefig(acc_plot_path)
            plt.close()

            logger.info(f"Training history plots saved: {loss_plot_path}, {acc_plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot training history: {e}")
            raise

    def plot_confusion_matrix(self) -> None:
        """
        Plots the confusion matrix.

        Returns:
            None
        """
        try:
            cm = confusion_matrix(self.y_test, self.y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            output_dir = os.path.join('artifacts', 'plots')
            os.makedirs(output_dir, exist_ok=True)
            cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_plot_path)
            plt.close()
            logger.info(f"Confusion matrix plot saved: {cm_plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            raise

    def plot_roc_curve(self) -> None:
        """
        Plots the ROC curve.

        Returns:
            None
        """
        try:
            fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid(True)
            output_dir = os.path.join('artifacts', 'plots')
            os.makedirs(output_dir, exist_ok=True)
            roc_plot_path = os.path.join(output_dir, 'roc_curve.png')
            plt.savefig(roc_plot_path)
            plt.close()
            logger.info(f"ROC curve plot saved: {roc_plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {e}")
            raise


class Trainer:
    """
    Manages the training process.

    Attributes:
        df (pd.DataFrame): The input data frame.
        target_column (str): The name of the target column.
    """

    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Initializes the Trainer with data and configurations.

        Args:
            df (pd.DataFrame): The input data frame.
            target_column (str): The name of the target column.
        """
        self.df = df
        self.target_column = target_column

    def run(self) -> None:
        """
        Runs the training process.

        Returns:
            None

        Raises:
            Exception: If an error occurs during training.
        """
        try:
            # Check GPU availability
            check_gpu()

            # Prepare data
            preprocessor = DataPreprocessor(self.df, self.target_column)
            X_train, X_test, y_train, y_test = preprocessor.prepare_data()

            # Handle imbalanced dataset
            class_weights = self.compute_class_weights(y_train)

            # Build and train model
            input_dim = X_train.shape[1]
            nn_model = NeuralNetworkModel(input_dim, class_weights)
            nn_model.hyperparameter_tuning(X_train, y_train)
            nn_model.train(X_train, y_train, X_test, y_test)

            # Evaluate model
            nn_model.evaluate(X_test, y_test)

            # Save model
            nn_model.save_model()

            # Plot training history and evaluation metrics
            nn_model.plot_training_history()
            nn_model.plot_confusion_matrix()
            nn_model.plot_roc_curve()

        except Exception as e:
            logger.error(f'An error occurred during training: {e}')
            raise

    def compute_class_weights(self, y: np.ndarray) -> dict:
        """
        Computes class weights for imbalanced datasets.

        Args:
            y (np.ndarray): Training labels.

        Returns:
            dict: Class weights.
        """
        class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights_dict = dict(enumerate(class_weights_array))
        logger.info(f"Computed class weights: {class_weights_dict}")
        return class_weights_dict


class CrossValidator:
    """
    Performs cross-validation.

    Attributes:
        df (pd.DataFrame): The input data frame.
        target_column (str): The name of the target column.
        n_splits (int): Number of folds in K-Fold cross-validation.
        random_state (int): Controls the shuffling applied to the data before splitting.
    """

    def __init__(self, df: pd.DataFrame, target_column: str, n_splits: int = 5, random_state: int = 42):
        """
        Initializes the CrossValidator with data and configurations.

        Args:
            df (pd.DataFrame): The input data frame.
            target_column (str): The name of the target column.
            n_splits (int, optional): Number of folds. Defaults to 5.
            random_state (int, optional): Random state for shuffling. Defaults to 42.
        """
        self.df = df
        self.target_column = target_column
        self.n_splits = n_splits
        self.random_state = random_state

    def run(self) -> None:
        """
        Runs cross-validation.

        Returns:
            None

        Raises:
            Exception: If an error occurs during cross-validation.
        """
        try:
            X = self.df.drop(self.target_column, axis=1)
            y = self.df[self.target_column].values

            # Data validation: check for missing values
            if self.df.isnull().values.any():
                logger.warning("Missing values detected in the dataset. Imputing missing values.")
            else:
                logger.info("No missing values detected in the dataset.")

            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            fold_metrics = []

            for fold, (train_index, val_index) in enumerate(kfold.split(X)):
                logger.info(f'Fold {fold + 1}/{self.n_splits}')

                X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                # Create preprocessing pipeline
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                ])

                # Fit and transform training data
                X_train_fold_processed = pipeline.fit_transform(X_train_fold)

                # Only transform validation data (do not fit)
                X_val_fold_processed = pipeline.transform(X_val_fold)

                # Handle imbalanced dataset
                class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
                class_weights_dict = dict(enumerate(class_weights_array))
                logger.debug(f"Class weights for fold {fold + 1}: {class_weights_dict}")

                # Build and train model
                input_dim = X_train_fold_processed.shape[1]
                nn_model = NeuralNetworkModel(input_dim, class_weights_dict)
                nn_model.hyperparameter_tuning(X_train_fold_processed, y_train_fold)
                nn_model.train(X_train_fold_processed, y_train_fold, X_val_fold_processed, y_val_fold)

                # Evaluate model
                nn_model.evaluate(X_val_fold_processed, y_val_fold)
                loss, accuracy = nn_model.model.evaluate(X_val_fold_processed, y_val_fold, verbose=0)
                roc_auc = roc_auc_score(y_val_fold, nn_model.y_pred_proba)
                fold_metrics.append({
                    'loss': loss,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc
                })

                # Save model
                nn_model.save_model()

            # Calculate average metrics
            avg_loss = np.mean([m['loss'] for m in fold_metrics])
            avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
            avg_roc_auc = np.mean([m['roc_auc'] for m in fold_metrics])

            logger.info(f"\nAverage Validation Loss: {avg_loss:.4f}")
            logger.info(f"Average Validation Accuracy: {avg_accuracy:.4f}")
            logger.info(f"Average Validation ROC AUC: {avg_roc_auc:.4f}")

        except Exception as e:
            logger.error(f'An error occurred during cross-validation: {e}')
            raise


class Predictor:
    """
    Loads a trained model and preprocessing pipeline to make predictions on new data.

    Example:
        predictor = Predictor(model_path='model.h5', pipeline_path='pipeline.pkl')
        new_sample = np.array([0.5, 1.2, 3.4])
        prediction = predictor.predict(new_sample)
    """

    def __init__(self, model_path: str, pipeline_path: str):
        """
        Initializes the Predictor with the model and preprocessing pipeline.

        Args:
            model_path (str): Path to the trained model file.
            pipeline_path (str): Path to the preprocessing pipeline file.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.pipeline = joblib.load(pipeline_path)
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Pipeline loaded from {pipeline_path}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model or pipeline: {e}")
            raise

    def predict(self, new_sample: np.ndarray) -> int:
        """
        Predicts the class of a new sample.

        Args:
            new_sample (np.ndarray): New sample data as a 1D array.

        Returns:
            int: Predicted class label.

        Raises:
            Exception: If an error occurs during prediction.
        """
        try:
            new_sample_processed = self.pipeline.transform(new_sample.reshape(1, -1))
            prediction_proba = self.model.predict(new_sample_processed).ravel()[0]
            prediction_class = int(prediction_proba > 0.5)
            logger.info(f'Prediction probability: {prediction_proba:.4f}')
            logger.info(f'Predicted class: {prediction_class}')
            return prediction_class
        except Exception as e:
            logger.error(f'An error occurred during prediction: {e}')
            raise


def main() -> None:
    """
    Main function to run the script.

    Usage:
        python script.py --mode train --data_path data.csv

    Arguments:
        --mode: Mode to run. Options are 'train', 'cross_validate', 'predict'.
        --data_path: Path to the dataset CSV file (required for 'train' and 'cross_validate' modes).
        --model_path: Path to the trained model file (required for 'predict' mode).
        --pipeline_path: Path to the preprocessing pipeline file (required for 'predict' mode).
        --sample_data: Comma-separated feature values for prediction (required for 'predict' mode).

    Examples:
        Training mode:
            python script.py --mode train --data_path data.csv

        Cross-validation mode:
            python script.py --mode cross_validate --data_path data.csv

        Prediction mode:
            python script.py --mode predict --model_path model.h5 --pipeline_path pipeline.pkl --sample_data "0.5,1.2,3.4"

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Train and evaluate neural network.')

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'cross_validate', 'predict'],
                        help='Mode to run. Options are "train", "cross_validate", "predict".')

    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the dataset CSV file (required for "train" and "cross_validate" modes).')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model file (required for "predict" mode).')

    parser.add_argument('--pipeline_path', type=str, default=None,
                        help='Path to the preprocessing pipeline file (required for "predict" mode).')

    parser.add_argument('--sample_data', type=str, default=None,
                        help='Comma-separated feature values for prediction (required for "predict" mode).')

    parser.add_argument('--target_column', type=str, default='target',
                        help='Name of the target column in the dataset. Defaults to "target".')

    args = parser.parse_args()

    if args.mode == 'train':
        if args.data_path is None:
            logger.error('Data path must be specified for training.')
            sys.exit(1)
        try:
            df = pd.read_csv(args.data_path)
            trainer = Trainer(df, args.target_column)
            trainer.run()
        except FileNotFoundError as e:
            logger.error(f'Data file not found: {e}')
            sys.exit(1)
        except Exception as e:
            logger.error(f'An error occurred during training: {e}')
            sys.exit(1)

    elif args.mode == 'cross_validate':
        if args.data_path is None:
            logger.error('Data path must be specified for cross-validation.')
            sys.exit(1)
        try:
            df = pd.read_csv(args.data_path)
            cross_validator = CrossValidator(df, args.target_column)
            cross_validator.run()
        except FileNotFoundError as e:
            logger.error(f'Data file not found: {e}')
            sys.exit(1)
        except Exception as e:
            logger.error(f'An error occurred during cross-validation: {e}')
            sys.exit(1)

    elif args.mode == 'predict':
        if args.model_path is None or args.pipeline_path is None or args.sample_data is None:
            logger.error('Model path, pipeline path, and sample data must be specified for prediction.')
            sys.exit(1)
        try:
            predictor = Predictor(args.model_path, args.pipeline_path)
            try:
                sample_values = [float(x.strip()) for x in args.sample_data.split(',')]
                sample_array = np.array(sample_values)
                prediction = predictor.predict(sample_array)
                logger.info(f'Predicted class: {prediction}')
            except ValueError as e:
                logger.error(f'Invalid sample data provided: {e}')
                sys.exit(1)
        except Exception as e:
            logger.error(f'An error occurred during prediction: {e}')
            sys.exit(1)
    else:
        logger.error(f'Invalid mode selected: {args.mode}')
        sys.exit(1)


if __name__ == "__main__":
    set_seed(42)
    main()
