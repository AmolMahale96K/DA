import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Synthetic data generation
np.random.seed(0)
num_samples = 1000
mileage = np.random.randint(10000, 100000, size=num_samples)
age = np.random.randint(1, 20, size=num_samples)
price = 20000 + 100 * mileage - 500 * age + np.random.normal(0, 5000, size=num_samples)

# Create DataFrame
car_data = pd.DataFrame({'Mileage': mileage, 'Age': age, 'Price': price})

# Display the first few rows of the dataset
print(car_data.head())

# Split the dataset into features (X) and target variable (y)
X = car_data[['Mileage', 'Age']]
y = car_data['Price']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices')
plt.show()
