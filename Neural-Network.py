# Import necessary libraries
import os
import sys
import logging
import datetime
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.utils import class_weight

import joblib

# Set random seeds for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Check for GPU availability
def check_gpu():
    """
    Check if GPU is available and log the information.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU(s) available: {gpus}")
    else:
        logger.warning("No GPU available. Training will be performed on CPU.")

class DataPreprocessor:
    """
    Class for data preparation and preprocessing.
    """
    def __init__(self, df, target_column, test_size=0.2, random_state=42):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None

    def prepare_data(self):
        """
        Prepare data for model training and evaluation.
        Splits data into training and testing sets.
        """
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
        X_test_processed = self.pipeline.transform(X_test)

        # Save the pipeline
        self.save_pipeline()

        return X_train_processed, X_test_processed, y_train.values, y_test.values

    def save_pipeline(self):
        """
        Save the preprocessing pipeline.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        pipeline_dir = os.path.join('artifacts', 'pipelines')
        os.makedirs(pipeline_dir, exist_ok=True)
        pipeline_filename = os.path.join(pipeline_dir, f'pipeline_{timestamp}.pkl')
        joblib.dump(self.pipeline, pipeline_filename)
        logger.info(f"Pipeline saved as {pipeline_filename}")

class NeuralNetworkModel:
    """
    Class for creating, training, and evaluating the neural network model.
    """
    def __init__(self, input_dim, class_weights=None):
        self.input_dim = input_dim
        self.model = None
        self.class_weights = class_weights

    def build_model(self, hp):
        """
        Build the Keras model with hyperparameters.
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

    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using Keras Tuner.
        """
        logger.info("Starting hyperparameter tuning...")

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

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        callbacks = [early_stopping, reduce_lr]

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )

        self.history = history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        """
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

    def save_model(self):
        """
        Save the trained model.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join('artifacts', 'models', timestamp)
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, 'model.h5')
        self.model.save(model_filename)
        logger.info(f"Model saved as {model_filename}")

    def plot_training_history(self):
        """
        Plot training and validation loss and accuracy over epochs.
        """
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

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        output_dir = os.path.join('artifacts', 'plots')
        os.makedirs(output_dir, exist_ok=True)
        cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
        logger.info(f"Confusion matrix plot saved: {cm_plot_path}")

    def plot_roc_curve(self):
        """
        Plot the ROC curve.
        """
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

class Trainer:
    """
    Class to manage the training process.
    """
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def run(self):
        """
        Run the training process.
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

    def compute_class_weights(self, y):
        """
        Compute class weights for imbalanced datasets.
        """
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights_dict = dict(enumerate(class_weights))
        logger.info(f"Computed class weights: {class_weights_dict}")
        return class_weights_dict

class CrossValidator:
    """
    Class to perform cross-validation.
    """
    def __init__(self, df, target_column, n_splits=5, random_state=42):
        self.df = df
        self.target_column = target_column
        self.n_splits = n_splits
        self.random_state = random_state

    def run(self):
        """
        Run cross-validation.
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
                X_val_fold_processed = pipeline.transform(X_val_fold)

                # Handle imbalanced dataset
                class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
                class_weights_dict = dict(enumerate(class_weights))

                # Build and train model
                input_dim = X_train_fold_processed.shape[1]
                nn_model = NeuralNetworkModel(input_dim, class_weights_dict)
                nn_model.hyperparameter_tuning(X_train_fold_processed, y_train_fold)
                nn_model.train(X_train_fold_processed, y_train_fold, X_val_fold_processed, y_val_fold)

                # Evaluate model
                nn_model.evaluate(X_val_fold_processed, y_val_fold)
                fold_metrics.append({
                    'loss': nn_model.model.evaluate(X_val_fold_processed, y_val_fold, verbose=0)[0],
                    'accuracy': nn_model.model.evaluate(X_val_fold_processed, y_val_fold, verbose=0)[1],
                    'roc_auc': roc_auc_score(y_val_fold, nn_model.y_pred_proba)
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
    Class for making predictions on new data.
    """
    def __init__(self, model_path, pipeline_path):
        self.model = tf.keras.models.load_model(model_path)
        self.pipeline = joblib.load(pipeline_path)

    def predict(self, new_sample):
        """
        Predict the class of a new sample.
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

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate neural network.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'cross_validate', 'predict'],
                        help='Mode to run: train, cross_validate, predict')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the dataset CSV file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model')
    parser.add_argument('--pipeline_path', type=str, default=None, help='Path to the preprocessing pipeline')
    parser.add_argument('--sample_data', type=str, default=None, help='New sample data for prediction')
    args = parser.parse_args()

    if args.mode == 'train':
        if args.data_path is None:
            logger.error('Data path must be specified for training.')
            sys.exit(1)
        df = pd.read_csv(args.data_path)
        trainer = Trainer(df, 'target')
        trainer.run()

    elif args.mode == 'cross_validate':
        if args.data_path is None:
            logger.error('Data path must be specified for cross-validation.')
            sys.exit(1)
        df = pd.read_csv(args.data_path)
        cross_validator = CrossValidator(df, 'target')
        cross_validator.run()

    elif args.mode == 'predict':
        if args.model_path is None or args.pipeline_path is None or args.sample_data is None:
            logger.error('Model path, pipeline path, and sample data must be specified for prediction.')
            sys.exit(1)
        predictor = Predictor(args.model_path, args.pipeline_path)
        sample_array = np.array([float(x) for x in args.sample_data.split(',')])
        predictor.predict(sample_array)

if __name__ == "__main__":
    main()
