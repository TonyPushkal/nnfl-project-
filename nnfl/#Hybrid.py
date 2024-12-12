import pandas as pd
import numpy as np
from tensorflow.keras.layers import InputLayer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
csv_path = 'C:\\Users\\Tonyp\\OneDrive\\Desktop\\New folder\\nnfl\\jena_climate_2009_2016.csv'
df = pd.read_csv(csv_path)

# Downsample the data to reduce redundancy
df = df[5::6]
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
temp = df['T (degC)']

# Normalize the data
scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(temp.values.reshape(-1, 1))

# Define a function to create sequences for supervised learning
def df_to_X_y(df, window_size=7):
    df_as_np = df
    X, y = [], []
    for i in range(len(df_as_np) - window_size):
        X.append(df_as_np[i:i + window_size])
        y.append(df_as_np[i + window_size])
    return np.array(X), np.array(y)

# Create input-output pairs
WINDOW_SIZE = 7
X, y = df_to_X_y(temp_scaled, WINDOW_SIZE)

# Split the data into training, validation, and test sets
X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]

# Add the required shape for CNN and LSTM models
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the Hybrid CNN-LSTM model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Single output for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

# Set callbacks
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Evaluate the model on the test set
test_predictions = model.predict(X_test).flatten()
rmse_test = mean_squared_error(y_test, test_predictions)
mae_test = mean_absolute_error(y_test, test_predictions)
mse_test = mean_squared_error(y_test, test_predictions)

print(f"Test Mean Square Error: {mse_test}")
print(f"Test Mean Aboslute Error: {mae_test}")
print(f"Test Root Mean Square Error: {rmse_test}")

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
fold_metrics = []

#for train_index, test_index in kf.split(X):
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]
    
    # Create the model (reinitialize for each fold)
   # model = Sequential([
        #Conv1D(64, kernel_size=2, activation='relu'),
       # Flatten(),
       # Dense(8, activation='relu'),
       # Dense(2, activation='linear')
   # ])
   # model.compile(
       # loss='mean_squared_error',
      #  optimizer=Adam(learning_rate=0.0001),
      #  metrics=['root_mean_squared_error']
  #  )
    
    # Train the model
   ## model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Evaluate the model
   # loss, rmse = model.evaluate(X_test, y_test, verbose=0)
    #fold_metrics.append({'loss': loss, 'rmse': rmse})
    #print(f"Fold Loss: {loss}, Fold RMSE: {rmse}")

# Calculate and display average metrics
#average_loss = np.mean([metric['loss'] for metric in fold_metrics])
#verage_rmse = np.mean([metric['rmse'] for metric in fold_metrics])
#print(f"Average Loss across folds: {average_loss}")
#print(f"Average RMSE across folds: {average_rmse}")


# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(test_predictions[:100], label='Predictions', color='blue')
plt.plot(y_test[:100], label='Actuals', color='orange')
plt.xlabel('Normalized Time')  # Label for x-axis
plt.ylabel('Temperature(Normalized)')  # Label for y-axis
plt.title("Test Predictions vs Actuals")
plt.legend()
plt.show()

# Visualize training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


