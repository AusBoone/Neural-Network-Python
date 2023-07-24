from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assume you have a DataFrame `df` with a binary target column 'target'
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())  # Add Batch Normalization layer
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())  # Add Batch Normalization layer
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.001)  # Define optimizer with learning rate
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Visualize the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Train the model for a given number of epochs
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr_on_plateau])

# Load the best model
model = load_model('best_model.h5')

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Use the model to predict the class of a new sample (you'll need to replace this with your own data)
new_sample = np.array([0]*X_train_scaled.shape[1])  # This is just a placeholder
new_sample_scaled = scaler.transform(new_sample.reshape(1, -1))
prediction = model.predict(new_sample_scaled)
print(f'Prediction for the new sample: {prediction}')

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kfold.split(X_train_scaled):
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping, model_checkpoint, reduce_lr_on_plateau])
