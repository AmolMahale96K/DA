import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data for sales and independent variables
np.random.seed(42)
num_samples = 1000
sales = np.random.normal(loc=1000, scale=200, size=num_samples)  # Mean sales: $1000, Standard deviation: $200
advertising_cost = np.random.normal(loc=200, scale=50, size=num_samples)  # Mean advertising cost: $200, Standard deviation: $50
price = np.random.normal(loc=50, scale=10, size=num_samples)  # Mean price: $50, Standard deviation: $10

# Create a DataFrame
data = pd.DataFrame({'Sales': sales, 'Advertising_Cost': advertising_cost, 'Price': price})

# Independent variables (advertising cost and price) and target variable (sales)
X = data[['Advertising_Cost', 'Price']]
y = data['Sales']

# Split the data into training and testing sets
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split the training set into 70% training and 30% validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.3, random_state=42)

# Print the shapes of training, validation, and testing sets
print("Training set shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("\nValidation set shapes:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("\nTesting set shapes:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Build a simple linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the validation data
y_val_pred = model.predict(X_val)

# Calculate mean squared error on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
print("\nMean Squared Error on Validation Set:", mse_val)

# Predict on the testing data
y_test_pred = model.predict(X_test)

# Calculate mean squared error on the testing set
mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error on Testing Set:", mse_test)

# Print coefficients and intercept
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)
