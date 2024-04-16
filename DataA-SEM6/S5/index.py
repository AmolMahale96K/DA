import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris_df = pd.read_csv("C:/Users/mahal/Desktop/iris.csv")

setosa_stats = iris_df[iris_df['variety'] == 'Setosa'].describe()
versicolor_stats = iris_df[iris_df['variety'] == 'Versicolor'].describe()
virginica_stats = iris_df[iris_df['variety'] == 'Virginica'].describe()

print("Basic Statistical Details for 'Iris-setosa':")
print(setosa_stats)

print("\nBasic Statistical Details for 'Iris-versicolor':")
print(versicolor_stats)

print("\nBasic Statistical Details for 'Iris-virginica':")
print(virginica_stats)

X = iris_df.drop('variety', axis=1)
y = iris_df['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the logistic regression model:", accuracy)
