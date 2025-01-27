import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset (replace with your actual file name)
df = pd.read_csv("USA Housing Dataset.csv") 

# Assuming 'price' is the target variable and 'bedrooms' is a relevant feature
X = df[['bedrooms']]  # Select the 'bedrooms' column as the feature
y = df['price']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate residuals (errors)
residuals = y_test - y_pred

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)  # Average error in dollars
mse = mean_squared_error(y_test, y_pred)  # Squared error in dollars
rmse = np.sqrt(mse)  # Root Mean Squared Error in dollars
r2_score = model.score(X_test, y_test)  # R-squared value for the model

# Output the results for house price predictions
print(f"Model Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")  # Displaying in dollars
print(f"Mean Squared Error (MSE): ${mse:.2f}")  # Displaying in dollars
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")  # In dollars
print(f"R-squared Value: {r2_score:.2f}")

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the data points
plt.scatter(X_test, y_test, color='blue', label='Data (House Prices)')

# Plot the regression line
plt.plot(X_test, y_pred, color='black', label='Linear Regression Model')

# Plot the residuals as vertical lines
for i in range(len(X_test)):
    plt.plot([X_test.iloc[i, 0], X_test.iloc[i, 0]], 
             [y_test.iloc[i], y_pred[i]], color='red', linestyle='--', label='Error' if i == 0 else "")

# Add labels and title
plt.xlabel('Number of Bedrooms')  # Replace with the actual feature name
plt.ylabel('Price (in dollars)')
plt.title('Linear Regression Model for Housing Prices')
plt.legend()

# Show the plot
plt.show()