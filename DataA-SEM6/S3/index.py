import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

n_users = 100
user_id = np.arange(1, n_users + 1)
gender = np.random.choice(['Male', 'Female'], size=n_users)
age = np.random.randint(18, 65, size=n_users)
estimated_salary = np.random.randint(20000, 100000, size=n_users)
purchased = np.random.choice([0, 1], size=n_users)

user_data = pd.DataFrame({
    'User ID': user_id,
    'Gender': gender,
    'Age': age,
    'Estimated Salary': estimated_salary,
    'Purchased': purchased
})

print("User dataset:")
print(user_data.head())

X = user_data[['Age', 'Estimated Salary']]
y = user_data['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("\nTraining set - Independent variables:")
print(X_train[:5])
print("\nTraining set - Target variable:")
print(y_train[:5])

print("\nTesting set - Independent variables:")
print(X_test[:5])
print("\nTesting set - Target variable:")
print(y_test[:5])

model = LogisticRegression().fit(X_train, y_train)

print("\nCoefficients: ",model.coef_)
print("Intercept:", model.intercept_)
