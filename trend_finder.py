import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data (Example: GDP, Inflation, Interest Rates, Stock Prices)
# using "US Financial Indicators - 1974 to 2024"
# https://www.kaggle.com/datasets/abhishekb7/us-financial-indicators-1974-to-2024
data = pd.read_csv("main.csv")

# Check data structure
print(data.head())

# Select relevant features
features = ["Interest_Rate", "Inflation", "GDP", "Unemployment"]
target = "sp500"

X = data[features]
y = data[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict stock prices
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red", linestyle="dashed")
plt.legend()
plt.title("Stock Price Predictions vs. Actual")
plt.show()

