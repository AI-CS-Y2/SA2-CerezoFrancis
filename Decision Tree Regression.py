import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def visualize_tree_regression(X, y, max_depths=[2, 5]):
  """
  Visualizes decision tree regression with different max_depth values.

  Args:
      X: Feature data (numpy array or pandas DataFrame)
      y: Target variable (numpy array or pandas Series)
      max_depths: List of max_depth values to use for the decision trees
  """
  
  # Store performance metrics for each model
  results = {}

  plt.figure()
  plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="Data (House Prices)")

  for depth in max_depths:
      regr = DecisionTreeRegressor(max_depth=depth)
      regr.fit(X, y)
      X_test = np.arange(min(X), max(X), 0.01)[:, np.newaxis]
      y_pred = regr.predict(X_test)
      
      # Store metrics for each depth
      y_test_pred = regr.predict(X)  # Predictions on training data for metrics
      mae = mean_absolute_error(y, y_test_pred)
      mse = mean_squared_error(y, y_test_pred)
      rmse = np.sqrt(mse)
      r2_score = regr.score(X, y)
      
      results[depth] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2_score}
      
      # Plot the regression line
      plt.plot(X_test, y_pred, label=f"max_depth={depth}")

  # Output the results for house price predictions
  for depth in max_depths:
      print(f"Results for max_depth={depth}:")
      print(f"  Mean Absolute Error (MAE): ${results[depth]['MAE']:.2f}")  # Displaying in dollars
      print(f"  Mean Squared Error (MSE): ${results[depth]['MSE']:.2f}")  # Displaying in dollars
      print(f"  Root Mean Squared Error (RMSE): ${results[depth]['RMSE']:.2f}")  # In dollars
      print(f"  R-squared Value: {results[depth]['R2']:.2f}")
      print()

  # Final plot settings
  plt.xlabel("Number of Bedrooms")  # or another feature that you are using
  plt.ylabel("Price (in dollars)")
  plt.title("Decision Tree Regression on Housing Prices")
  plt.legend()
  plt.show()

# Load your dataset
df = pd.read_csv('USA Housing Dataset.csv')  # Replace with the actual path to your dataset

# Assuming 'bedrooms' is the numerical feature column for prediction
X = df['bedrooms'].to_numpy().reshape(-1, 1)  # Reshape to 2D array
y = df['price']

# Visualize the decision tree regression with different max_depths
visualize_tree_regression(X, y, max_depths=[2, 5])
