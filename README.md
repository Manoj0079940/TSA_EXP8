# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# === 1. Read the dataset ===
data = pd.read_csv("AirPassengers.csv")

# Display column names to confirm structure
print("Columns in dataset:", data.columns.tolist())

# Automatically detect date column
date_col = None
for c in data.columns:
    if "month" in c.lower() or "date" in c.lower() or "time" in c.lower():
        date_col = c
        break

if date_col is None:
    raise ValueError("No date-like column found in dataset!")

# Convert date column to datetime and set as index
data[date_col] = pd.to_datetime(data[date_col])
data = data.set_index(date_col).sort_index()

# Select a numeric column (first non-date column)
target_col = data.select_dtypes(include=[np.number]).columns[0]
print(f"Using '{target_col}' as target column.\n")

# Focus on that column
target_data = data[[target_col]]

# === 2. Display shape and first 10 rows ===
print("Shape of the dataset:", target_data.shape)
print("\nFirst 10 rows of the dataset:")
print(target_data.head(10))

# === 3. Plot Original Data ===
plt.figure(figsize=(12, 6))
plt.plot(target_data, label=f'Original {target_col} Data', color='blue')
plt.title('Original Time Series Data (AirPassengers)')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.grid()
plt.show()

# === 4. Moving Average ===
rolling_mean_5 = target_data[target_col].rolling(window=5).mean()
rolling_mean_10 = target_data[target_col].rolling(window=10).mean()

print("\nFirst 10 values of rolling mean (window=5):")
print(rolling_mean_5.head(10))
print("\nFirst 20 values of rolling mean (window=10):")
print(rolling_mean_10.head(20))

# Plot Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(target_data[target_col], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)', color='orange')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', color='green')
plt.title('Moving Average of AirPassengers Data')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.legend()
plt.grid()
plt.show()

# === 5. Data Transformation for Modeling ===
# Resample monthly (already monthly, but to ensure consistency)
data_monthly = target_data.resample('MS').mean()

# Scale data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

# Exponential smoothing needs strictly positive data for multiplicative models
scaled_data = scaled_data + 1e-3

# === 6. Train-test split ===
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# === 7. Exponential Smoothing Model ===
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot train, test, and predictions
ax = train_data.plot(label='Train Data', figsize=(12, 6))
test_data.plot(ax=ax, label='Test Data')
test_predictions_add.plot(ax=ax, label='Predictions')
ax.legend()
ax.set_title('Exponential Smoothing - AirPassengers Forecast')
plt.show()

# Inverse transform both test and predictions
test_data_original = pd.Series(
    scaler.inverse_transform(test_data.values.reshape(-1, 1)).flatten(),
    index=test_data.index
)

predictions_original = pd.Series(
    scaler.inverse_transform(test_predictions_add.values.reshape(-1, 1)).flatten(),
    index=test_predictions_add.index
)

# Compute RMSE on original (non-normalized) values
rmse_original = np.sqrt(mean_squared_error(test_data_original, predictions_original))
print(f"Root Mean Squared Error (RMSE) on Original Data: {rmse_original:.4f}")


# Evaluate RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# === 8. Forecast for next 1/4th of data ===
model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
steps = int(len(data_monthly) / 4)
predictions = model_full.forecast(steps=steps)

# Plot forecast
ax = scaled_data.plot(label='Historical Data', figsize=(12, 6))
predictions.plot(ax=ax, label='Forecast', color='red')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Value')
ax.set_title('Forecast for Next Quarter - AirPassengers')
plt.grid()
plt.show()
```
### OUTPUT:


<img width="1266" height="660" alt="image" src="https://github.com/user-attachments/assets/f6c14131-b4d5-4a70-9914-f6b140ea84f5" />

<img width="1275" height="679" alt="image" src="https://github.com/user-attachments/assets/22346bd2-4c75-4c64-bf89-203e3eaf109c" />


<img width="1253" height="675" alt="image" src="https://github.com/user-attachments/assets/78c07b30-1c5f-4f1e-8a67-d3b7c9ecf9f1" />
<img width="1264" height="686" alt="image" src="https://github.com/user-attachments/assets/9cda44ff-ff33-4b48-81e8-4073dcd126cd" />



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
