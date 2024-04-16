import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data for real estate prices and independent variables
np.random.seed(42)
num_samples = 1000
square_footage = np.random.randint(1000, 3000, size=num_samples)
num_bedrooms = np.random.randint(2, 6, size=num_samples)
num_bathrooms = np.random.randint(2, 4, size=num_samples)
garage_capacity = np.random.randint(1, 3, size=num_samples)
year_built = np.random.randint(1980, 2020, size=num_samples)

# Generate synthetic prices based on the independent variables
price = 1000 * square_footage + 5000 * num_bedrooms + 3000 * num_bathrooms + 2000 * garage_capacity - 1000 * (2022 - year_built)

# Create a DataFrame
data = pd.DataFrame({
    'Square_Footage': square_footage,
    'Num_Bedrooms': num_bedrooms,
    'Num_Bathrooms': num_bathrooms,
    'Garage_Capacity': garage_capacity,
    'Year_Built': year_built,
    'Price': price
})

# Independent variables (features) and target variable (price)
X = data[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Garage_Capacity', 'Year_Built']]
y = data['Price']

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
