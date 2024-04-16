import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)

total = np.random.uniform(1000, 5000, 100)
sales = 50 + 3 * total + np.random.normal(0, 100, 100)

sd = pd.DataFrame({'Total': total, 'Sales': sales})

print(sd.head())

X_train, X_test, y_train, y_test = train_test_split(sd[['Total']], sd['Sales'],train_size=0.7,test_size=0.3, random_state=0)

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




