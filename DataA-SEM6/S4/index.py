import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)

length = np.random.uniform(10, 100, 100)  
weight = 50 + 3 * length + np.random.normal(0, 10, 100) 

fish_data = pd.DataFrame({'Length': length, 'Weight': weight})

print(fish_data.head())

X_train, X_test, y_train, y_test = train_test_split(fish_data[['Length']], fish_data['Weight'], test_size=0.3, random_state=0)

print("\nTraining set - Independent variable (Length):")
print(X_train[:5])
print("\nTraining set - Target variable (Weight):")
print(y_train[:5])

print("\nTesting set - Independent variable (Length):")
print(X_test[:5])
print("\nTesting set - Target variable (Weight):")
print(y_test[:5])

model = LinearRegression().fit(X_train, y_train)

print("\nCoefficients (slope):", model.coef_)
print("Intercept:", model.intercept_)
