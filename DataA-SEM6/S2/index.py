import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)

property_size = np.random.uniform(500, 5000, 100)
price = 10000 + 10 * property_size + np.random.normal(0, 1000, 100)

realestate_data = pd.DataFrame({'Property Size': property_size, 'Price': price})

print(realestate_data.head())

X_train, X_test, y_train, y_test = train_test_split(realestate_data[['Property Size']], realestate_data['Price'], test_size=0.3, random_state=0)

print("\nTraining set - Independent variable:")
print(X_train[:5])
print("\nTraining set - Target variable:")
print(y_train[:5])

print("\nTesting set - Independent variable:")
print(X_test[:5])
print("\nTesting set - Target variable:")
print(y_test[:5])

model = LinearRegression().fit(X_train, y_train)

print("\nCoefficients (slope):", model.coef_)
print("Intercept:", model.intercept_)
