import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Creating a transactions dataset
transactions = [
    {'TransactionID': 1, 'Product': 'Milk', 'Quantity': 2, 'Price': 1.5},
    {'TransactionID': 2, 'Product': 'Bread', 'Quantity': 1, 'Price': 2},
    {'TransactionID': 3, 'Product': 'Eggs', 'Quantity': 12, 'Price': 0.2},
    {'TransactionID': 4, 'Product': 'Milk', 'Quantity': 1, 'Price': 1.5},
    {'TransactionID': 5, 'Product': 'Bread', 'Quantity': 3, 'Price': 2},
    {'TransactionID': 6, 'Product': 'Eggs', 'Quantity': 6, 'Price': 0.2},
    {'TransactionID': 7, 'Product': 'Milk', 'Quantity': 2, 'Price': 1.5},
    {'TransactionID': 8, 'Product': 'Bread', 'Quantity': 1, 'Price': 2},
    {'TransactionID': 9, 'Product': 'Eggs', 'Quantity': 12, 'Price': 0.2},
    {'TransactionID': 10, 'Product': 'Milk', 'Quantity': 1, 'Price': 1.5},
]

# Convert transactions into a DataFrame
df = pd.DataFrame(transactions)

# Independent variable (quantity) and target variable (total price)
X = df[['Quantity']]
y = df['Price'] * df['Quantity']

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
print("\nCoefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
