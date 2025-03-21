# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset (replace with your dataset file)
# Example: Heart Disease UCI dataset from Kaggle
data = pd.read_csv('heart_disease_data.csv')

# Step 2: Preprocess the data
X = data.drop(columns=['target'])  # Features
y = data['target']  # Labels (1: Disease, 0: No disease)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of the model:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Save the trained model (for deployment)
import pickle
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'heart_disease_model.pkl'")

