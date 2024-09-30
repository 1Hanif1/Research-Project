import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load labeled data (features and labels)
data = np.load("labeled_data.npy", allow_pickle=True).item()
features = data["features"]
labels = data["labels"]

# Check label distribution
print(f"Total label distribution: {np.bincount(labels)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Apply SMOTE to the training data to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the new label distribution after SMOTE
print(f"Resampled label distribution: {np.bincount(y_train_resampled)}")

# Train the SVM model on the resampled data
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_resampled, y_train_resampled)

# Test the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(svm_model, 'svm_model.pkl')
print("Model saved to svm_model.pkl")
