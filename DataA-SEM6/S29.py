import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the student dataset
student_data = pd.read_csv('student_dataset.csv')  # Replace 'student_dataset.csv' with the actual filename

# Display the first few rows of the dataset
print(student_data.head())

# Preprocess the data if necessary

# Split the dataset into features (X) and target variable (y)
X = student_data.drop(columns=['target_column'])  # Replace 'target_column' with the name of the target variable
y = student_data['target_column']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
