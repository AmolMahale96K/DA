import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the number of users
num_users = 1000

# Generate synthetic data for users
user_ids = np.arange(1, num_users + 1)
ages = np.random.randint(18, 80, size=num_users)
genders = np.random.choice(['Male', 'Female'], size=num_users)
income = np.random.randint(20000, 100000, size=num_users)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=num_users)
# Generate synthetic data for the target variable (e.g., whether a user purchased a product)
# For demonstration purposes, let's assume the target variable is binary (0 or 1)
purchased = np.random.choice([0, 1], size=num_users)

# Create a DataFrame for the user dataset
user_data = pd.DataFrame({
    'UserID': user_ids,
    'Age': ages,
    'Gender': genders,
    'Income': income,
    'Education': education,
    'Purchased': purchased
})

# Preprocess the data: Convert categorical variables into dummy/indicator variables
user_data_encoded = pd.get_dummies(user_data, columns=['Gender', 'Education'], drop_first=True)

# Split the dataset into features (independent variables) and the target variable
X = user_data_encoded.drop(columns=['UserID', 'Purchased'])
y = user_data_encoded['Purchased']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
