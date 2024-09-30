import numpy as np
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.metrics import accuracy_score
import joblib

# Load the saved model from the file
svm_model = joblib.load('svm_model.pkl')
print("Model loaded from svm_model.pkl")

# Load labeled data (features and labels)
data = np.load("labeled_data.npy", allow_pickle=True).item()
features = data["features"]
labels = data["labels"]

# Split the data into training and testing sets (same as before)
_, X_test, _, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Use the loaded model to make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy using loaded model: {accuracy * 100:.2f}%")
