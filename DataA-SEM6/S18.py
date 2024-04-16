import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
heights = np.random.normal(loc=160, scale=10, size=100)  # Mean height: 160 cm, Standard deviation: 10 cm
weights = 0.6 * heights + np.random.normal(loc=0, scale=5, size=100)  # Weight = 0.6*height + random noise

# Create a DataFrame
data = pd.DataFrame({'Height': heights, 'Weight': weights})

# Independent variable (height) and target variable (weight)
X = data[['Height']]
y = data['Weight']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of training and testing sets
print("Training set shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("\nTesting set shapes:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Build a simple linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Print coefficients and intercept
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)
