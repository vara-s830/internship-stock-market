import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# --- 1. Data Loading and Preprocessing ---
# In a real-world scenario, you would load your data from a CSV file like this:
# dataset = pd.read_csv('Google_Stock_Price_Test.csv')
# The following code generates dummy data for demonstration purposes.

print("Generating dummy stock data...")
# Generate a series that loosely resembles a stock price trend
np.random.seed(42)
days = 126
start_price = 1500
volatility = 0.015
trend = np.linspace(0.01, 0.05, days) # A slight upward trend
noise = np.random.normal(0, volatility, days)
dummy_prices = start_price * np.exp(np.cumsum(trend + noise))

# Create a DataFrame with a 'Close' column
dataset_train = pd.DataFrame(dummy_prices[:100], columns=['Close'])
dataset_test = pd.DataFrame(dummy_prices[100:], columns=['Close'])

# Visualize the dummy data to confirm its shape
plt.figure(figsize=(10, 6))
plt.plot(dataset_train['Close'], color='blue', label='Training Data')
plt.plot(range(100, len(dummy_prices)), dataset_test['Close'], color='red', label='Test Data')
plt.title('Dummy Stock Price Data')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Use the 'Close' column for our model
training_set = dataset_train.iloc[:, 0:1].values
test_set = dataset_test.iloc[:, 0:1].values

print("Dummy data generated and visualized.")

# --- 2. Feature Scaling ---
# Scale the data to a range of 0 to 1 for the LSTM model.
# This improves performance and stability.
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

print("Data has been scaled using MinMaxScaler.")

# --- 3. Preparing Datasets for Training ---
# LSTMs need a 3D input (samples, timesteps, features).
# We'll use the last 60 days of data to predict the next day's price.
X_train = []
y_train = []
timesteps = 60
for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data to the required 3D format for the LSTM layer
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(f"Training data reshaped to: {X_train.shape}")

# --- 4. Model Development ---
# Build the LSTM model as described in the report.
regressor = Sequential()

# First LSTM layer with Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Second LSTM layer with Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Third LSTM layer with Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Fourth (and final) LSTM layer with Dropout regularization
# We set return_sequences=False as this is the last LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# The output layer (Dense layer)
regressor.add(Dense(units=1))

# Compile the model
regressor.compile(optimizer='adam', loss='mean_squared_error')
print("Model compiled successfully.")

# --- 5. Model Training ---
print("Training the model...")
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
print("Model training complete.")

# --- 6. Predicting the Output ---
# First, prepare the test data in the same way as the training data.
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price_scaled = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Predictions complete.")

# --- 7. Result Visualization ---
# Plot the actual vs predicted prices to evaluate the model's performance.
plt.figure(figsize=(10, 6))
plt.plot(test_set, color='red', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print("Visualization complete.")

