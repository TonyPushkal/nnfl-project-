import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the data
file_path = 'C:/Users/tonyp/Downloads/DailyDelhiClimateTest.csv'
  
data = pd.read_csv(file_path)

# Convert 'date' to datetime format and set it as the index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Replace outliers if needed (example for 'meanpressure')
median_pressure = data['meanpressure'].median()
data.loc[data['meanpressure'] < 900, 'meanpressure'] = median_pressure

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Continue with your sequence creation, model building, etc.


# Creating sequences for multi-output prediction
def create_sequences_multi_output(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])  # Predict all columns
    return np.array(x), np.array(y)

sequence_length = 10
x, y = create_sequences_multi_output(scaled_data, sequence_length)

# Splitting into training and testing sets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data to fit the LSTM model
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# Build the LSTM Model for Multi-Output Prediction
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=y_train.shape[1]))  # Output layer for all parameters

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Predict on the Test Set
predictions = model.predict(x_test)

# Inverse the scaling to get the actual values
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Evaluate the Model - Example for Temperature
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(y_test_actual[:, 0], color='blue', label='Actual Mean Temperature')
plt.plot(predictions[:, 0], color='red', label='Predicted Mean Temperature')
plt.title('Mean Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Similarly, you can plot for other parameters like humidity, wind_speed, and meanpressure
