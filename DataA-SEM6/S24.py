import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic data for real estate properties
num_properties = 1000
locations = np.random.choice(['City', 'Suburb', 'Rural'], size=num_properties)
sizes = np.random.randint(500, 5000, size=num_properties)  # in square feet
num_bedrooms = np.random.randint(1, 6, size=num_properties)
num_bathrooms = np.random.randint(1, 4, size=num_properties)
age = np.random.randint(1, 50, size=num_properties)  # age of the property in years
prices = 50000 + 100 * sizes + 20000 * num_bedrooms - 5000 * age  # generating synthetic prices

# Create a DataFrame for the real estate dataset
real_estate_data = pd.DataFrame({
    'Location': locations,
    'Size': sizes,
    'NumBedrooms': num_bedrooms,
    'NumBathrooms': num_bathrooms,
    'Age': age,
    'Price': prices
})

# Identify the independent variables (features) and the target variable
X = real_estate_data[['Size', 'NumBedrooms', 'NumBathrooms', 'Age']]
y = real_estate_data['Price']

# Split the dataset into training and testing sets (80% training, 20% testing)
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

# Print coefficients and intercept
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)
