# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Download stock data (Example: Apple - AAPL)
data = yf.download("AAPL", start="2018-01-01", end="2024-01-01")

# Use only Closing price
data = data[['Close']]

# Create feature (days)
data['Day'] = np.arange(len(data))

# Input and Output
X = data[['Day']]
y = data['Close']

# Train the ML model
model = LinearRegression()
model.fit(X, y)

# Predict next 10 days
future_days = np.arange(len(data), len(data) + 10).reshape(-1, 1)
predictions = model.predict(future_days)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(data['Day'], data['Close'], label="Actual Price")
plt.plot(future_days, predictions, color='red', label="Predicted Price")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction using ML")
plt.legend()
plt.show()

# Print predicted prices
print("Predicted prices for next 10 days:")
for i, price in enumerate(predictions):
    print(f"Day {i+1}: ${price:.2f}")
