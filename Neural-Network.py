# Import necessary libraries
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

# Example DataFrame
df = pd.DataFrame(np.random.rand(1000, 20), columns=[f'feature_{i}' for i in range(20)])
df['target'] = np.random.randint(2, size=1000)

def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """
    Prepare data for model training and evaluation.
    Splits data into training and testing sets, and scales features.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features to have mean=0 and variance=1
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def create_model(input_dim):
    """
    Create a Sequential model and add layers to it.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  # Add Batch Normalization layer
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  # Add Batch Normalization layer
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def compile_and_train_model(model, X_train, y_train, learning_rate=0.001, epochs=10, batch_size=32, validation_split=0.2):
    """
    Compile and train the model.
    """
    optimizer = Adam(learning_rate=learning_rate)  # Define optimizer with learning rate
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Print the model summary
    print(model.summary())

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    # Train the model for a given number of epochs
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping, model_checkpoint, reduce_lr_on_plateau])

    # Load the best model
    model = load_model('best_model.h5')

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')


def predict_new_sample(model, new_sample, scaler):
    """
    Use the model to predict the class of a new sample.
    """
    new_sample_scaled = scaler.transform(new_sample.reshape(1, -1))
    prediction = model.predict(new_sample_scaled)
    print(f'Prediction for the new sample: {prediction}')


def cross_validation(X_train, y_train, n_splits=5, random_state=42):
    """
    Perform KFold cross-validation.
    """
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    # Define the K-Fold cross-validator. The number of folds is set by n_splits.
    # shuffle=True means the data will be shuffled before being split into folds.
    # random_state is set for reproducibility.
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # kfold.split(X_train) generates indices to split data into training and validation set.
    for train_index, val_index in kfold.split(X_train):
        # Use the indices to split the data into training and validation sets for both features and target
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Create a new instance of the model for the current fold
        model = create_model(X_train_fold.shape[1])

        # Compile and train the model on the training data for this fold
        model = compile_and_train_model(model, X_train_fold, y_train_fold)

        # Evaluate the trained model on the validation data for this fold
        evaluate_model(model, X_val_fold, y_val_fold)


def main(df, target_column):
    """
    Main function to run the script.
    """
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, target_column)
    model = create_model(X_train.shape[1])
    model = compile_and_train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Example new sample
    new_sample = np.random.rand(X_train.shape[1])
    predict_new_sample(model, new_sample, scaler)

    cross_validation(X_train, y_train)

# Run the main function
main(df, 'target')
