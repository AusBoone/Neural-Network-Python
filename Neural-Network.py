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
    Splits data into training and testing sets.
    """
    X = df.drop(target_column, axis=1).values  # Convert to numpy array
    y = df[target_column].values

    # Split the dataset into training and testing sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_model(input_dim):
    """
    Create a Sequential model and add layers to it.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def compile_and_train_model(model, X_train, y_train, validation_data=None,
                            learning_rate=0.001, epochs=50, batch_size=32):
    """
    Compile and train the model.
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    callbacks = [early_stopping, reduce_lr_on_plateau]

    # Train the model
    if validation_data is not None:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=validation_data, callbacks=callbacks, verbose=0)
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, callbacks=callbacks, verbose=0)

    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    return loss, accuracy

def predict_new_sample(model, new_sample, scaler):
    """
    Use the model to predict the class of a new sample.
    """
    new_sample_scaled = scaler.transform(new_sample.reshape(1, -1))
    prediction = model.predict(new_sample_scaled)
    print(f'Prediction for the new sample: {prediction[0][0]:.4f}')

def cross_validation(X, y, n_splits=5, random_state=42):
    """
    Perform KFold cross-validation.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X)):
        print(f'Fold {fold + 1}/{n_splits}')

        # Split the data
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Scale the data
        scaler = StandardScaler().fit(X_train_fold)
        X_train_fold_scaled = scaler.transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        # Create and train the model
        model = create_model(X_train_fold_scaled.shape[1])
        model, _ = compile_and_train_model(model, X_train_fold_scaled, y_train_fold,
                                           validation_data=(X_val_fold_scaled, y_val_fold))

        # Evaluate the model
        loss, accuracy = evaluate_model(model, X_val_fold_scaled, y_val_fold)
        fold_metrics.append({'loss': loss, 'accuracy': accuracy})

    # Calculate average metrics
    avg_loss = np.mean([m['loss'] for m in fold_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    print(f'\nAverage Validation Loss: {avg_loss:.4f}')
    print(f'Average Validation Accuracy: {avg_accuracy:.4f}')

def main(df, target_column):
    """
    Main function to run the script.
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, target_column)

    # Scale the data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the model
    model = create_model(X_train_scaled.shape[1])
    model, history = compile_and_train_model(model, X_train_scaled, y_train,
                                             validation_data=(X_test_scaled, y_test))

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Predict a new sample
    new_sample = np.random.rand(X_train_scaled.shape[1])
    predict_new_sample(model, new_sample, scaler)

    # Perform cross-validation
    print('\nStarting cross-validation...')
    cross_validation(X_train, y_train)

# Run the main function
if __name__ == "__main__":
    main(df, 'target')

