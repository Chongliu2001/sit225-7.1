import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load the data from the corrected CSV file
data = pd.read_csv('sensor_data_corrected_converted.csv')

# 2. Extract temperature and humidity data
# The temperature will be the independent variable (X), and humidity will be the dependent variable (y)
X = data['temperature'].values.reshape(-1, 1)
y = data['humidity'].values

# 3. Train a linear regression model
# The model will learn the relationship between temperature and humidity
model = LinearRegression()
model.fit(X, y)

# 4. Generate test data for temperature
# We create 100 equally spaced temperature values between the minimum and maximum observed temperatures
X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# 5. Predict humidity for the test temperature values
# The model will predict humidity based on the test temperatures
y_pred = model.predict(X_test)

# 6. Plot the original data and the regression line
# The scatter plot shows the actual temperature vs. humidity data points
# The red line represents the predicted trend line from the linear regression model
plt.scatter(X, y, label='Original Data')
plt.plot(X_test, y_pred, color='red', label='Trend Line')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.legend()
plt.show()

# 7. Analyze the trend line and outliers
# We can observe if the trend line follows the original data points and identify any outliers
# Outliers are data points that are significantly different from the trend line

# 8. Filter out some outliers and retrain the model
# We may choose to remove extreme temperature values that could be considered outliers
# Here, we remove data points with temperatures higher than a certain threshold, e.g., 24.5 degrees
filtered_data = data[data['temperature'] < 24.5]
X_filtered = filtered_data['temperature'].values.reshape(-1, 1)
y_filtered = filtered_data['humidity'].values

# Retrain the model with the filtered data
model.fit(X_filtered, y_filtered)

# Predict humidity for the test temperature values with the filtered model
y_pred_filtered = model.predict(X_test)

# Plot the filtered data and the new regression line
plt.scatter(X_filtered, y_filtered, label='Filtered Data')
plt.plot(X_test, y_pred_filtered, color='blue', label='Filtered Trend Line')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.legend()
plt.show()

# 9. Further analysis
# Repeat the filtering and retraining process as necessary, comparing the new trend lines with the original
