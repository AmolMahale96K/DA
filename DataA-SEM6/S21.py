import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration
np.random.seed(42)
num_samples = 100
feature_column = np.random.randint(1, 100, size=num_samples)
sales = 100 + 5 * feature_column + np.random.normal(0, 20, num_samples)

# Create a DataFrame
sales_data = pd.DataFrame({'Feature': feature_column, 'Sales': sales})

# Display the first few rows of the dataset
print(sales_data.head())

# Identify the independent variable (feature) and the target variable
X = sales_data[['Feature']]
y = sales_data['Sales']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the actual vs. predicted sales
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.show()
