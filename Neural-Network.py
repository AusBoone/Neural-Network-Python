# Import necessary libraries
import os
import joblib
import numpy as np
import pandas as pd
import logging
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
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

# Example DataFrame
df = pd.DataFrame(np.random.rand(1000, 20), columns=[f'feature_{i}' for i in range(20)])
df['target'] = np.random.randint(2, size=1000)

def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """
    Prepare data for model training and evaluation.
    Splits data into training and testing sets.
    Includes data validation checks.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Data validation: check for missing values
    if X.isnull().values.any() or y.isnull().values.any():
        logger.warning("Missing values detected in the dataset. Imputing missing values.")
        # Impute missing values with mean strategy
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    else:
        logger.info("No missing values detected in the dataset.")

    # Convert to numpy array
    X = X.values
    y = y.values

    # Split the dataset into training and testing sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_model(hp_units_input=64, hp_units_hidden=32, hp_dropout=0.5, hp_l2=0.01, learning_rate=0.001, input_dim=None):
    """
    Create a Sequential model with hyperparameters.
    """
    model = Sequential()
    model.add(Dense(hp_units_input, input_dim=input_dim, activation='relu', kernel_regularizer=l2(hp_l2)))
    model.add(BatchNormalization())
    model.add(Dropout(hp_dropout))
    model.add(Dense(hp_units_hidden, activation='relu', kernel_regularizer=l2(hp_l2)))
    model.add(BatchNormalization())
    model.add(Dropout(hp_dropout))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def compile_and_train_model(model, X_train, y_train, validation_data=None,
                            epochs=50, batch_size=32, callbacks=None):
    """
    Compile and train the model.
    """
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data, callbacks=callbacks, verbose=0)

    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f'Test loss: {loss:.4f}')
    logger.info(f'Test accuracy: {accuracy:.4f}')
    return loss, accuracy

def predict_new_sample(model, new_sample, scaler):
    """
    Use the model to predict the class of a new sample.
    """
    new_sample_scaled = scaler.transform(new_sample.reshape(1, -1))
    prediction = model.predict(new_sample_scaled)
    logger.info(f'Prediction for the new sample: {prediction[0][0]:.4f}')

def plot_training_history(history, output_dir='plots'):
    """
    Plot training and validation loss and accuracy over epochs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """
    logger.info("Starting hyperparameter tuning...")

    # Wrapper function for model creation
    def create_model_wrapper(hp_units_input=64, hp_units_hidden=32, hp_dropout=0.5, hp_l2=0.01, learning_rate=0.001):
        return create_model(hp_units_input=hp_units_input,
                            hp_units_hidden=hp_units_hidden,
                            hp_dropout=hp_dropout,
                            hp_l2=hp_l2,
                            learning_rate=learning_rate,
                            input_dim=X_train.shape[1])

    model = KerasClassifier(build_fn=create_model_wrapper, epochs=50, batch_size=32, verbose=0)

    # Define hyperparameter grid
    param_distribs = {
        'hp_units_input': [32, 64, 128],
        'hp_units_hidden': [16, 32, 64],
        'hp_dropout': [0.2, 0.5],
        'hp_l2': [0.0, 0.01, 0.1],
        'learning_rate': [1e-3, 1e-4],
    }

    # Randomized search over hyperparameters
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distribs,
                                       n_iter=5, cv=3, verbose=1, n_jobs=-1)
    random_search_result = random_search.fit(X_train, y_train)

    logger.info(f"Best parameters found: {random_search_result.best_params_}")
    logger.info(f"Best score: {random_search_result.best_score_:.4f}")

    return random_search_result.best_estimator_

def cross_validation(X, y, n_splits=5, random_state=42):
    """
    Perform KFold cross-validation.
    """
    logger.info("Starting cross-validation...")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X)):
        logger.info(f'Fold {fold + 1}/{n_splits}')

        # Split the data
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Scale the data
        scaler = StandardScaler().fit(X_train_fold)
        X_train_fold_scaled = scaler.transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        callbacks = [early_stopping, reduce_lr_on_plateau]

        # Hyperparameter tuning
        model = hyperparameter_tuning(X_train_fold_scaled, y_train_fold)

        # Evaluate the model
        loss, accuracy = evaluate_model(model, X_val_fold_scaled, y_val_fold)
        fold_metrics.append({'loss': loss, 'accuracy': accuracy})

        # Save the model with fold number and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_filename = f"model_fold{fold + 1}_{timestamp}.h5"
        model.model.save(model_filename)
        logger.info(f"Model saved as {model_filename}")

    # Calculate average metrics
    avg_loss = np.mean([m['loss'] for m in fold_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    logger.info(f"\nAverage Validation Loss: {avg_loss:.4f}")
    logger.info(f"Average Validation Accuracy: {avg_accuracy:.4f}")

def main(df, target_column):
    """
    Main function to run the script.
    """
    # Check GPU availability
    check_gpu()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, target_column)

    # Scale the data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    scaler_filename = f'scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_filename)
    logger.info(f"Scaler saved as {scaler_filename}")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callbacks = [early_stopping, reduce_lr_on_plateau]

    # Hyperparameter tuning
    model = hyperparameter_tuning(X_train_scaled, y_train)

    # Evaluate the model
    loss, accuracy = evaluate_model(model, X_test_scaled, y_test)

    # Save the model with timestamp
    model_filename = f"final_model_{timestamp}.h5"
    model.model.save(model_filename)
    logger.info(f"Final model saved as {model_filename}")

    # Plot training history
    if hasattr(model, 'history'):
        plot_training_history(model.history)
    else:
        logger.warning("No training history available for plotting.")

    # Predict a new sample
    new_sample = np.random.rand(X_train_scaled.shape[1])
    predict_new_sample(model, new_sample, scaler)

    # Perform cross-validation
    cross_validation(X_train, y_train)

# Run the main function
if __name__ == "__main__":
    main(df, 'target')
