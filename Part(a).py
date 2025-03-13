import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Paths to dataset
mask_path = "/home/harsh-d/Desktop/Face-Mask-Detection/dataset/with_mask"
without_mask_path = "/home/harsh-d/Desktop/Face-Mask-Detection/dataset/without_mask"

# Feature extraction function
def extract_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (64, 64))  # Resize for consistency

    # Compute HOG features
    hog_features = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Compute LBP features
    lbp = local_binary_pattern(img_resized, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # Concatenate features
    features = np.hstack((hog_features, lbp_hist))
    return features

# Load images and extract features
X, y = [], []
for file in os.listdir(mask_path):
    img = cv2.imread(os.path.join(mask_path, file))
    if img is not None:
        X.append(extract_features(img))
        y.append(1)  # Label: With Mask (1)

for file in os.listdir(without_mask_path):
    img = cv2.imread(os.path.join(without_mask_path, file))
    if img is not None:
        X.append(extract_features(img))
        y.append(0)  # Label: Without Mask (0)

# Convert to NumPy array
X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Feature extraction complete. Training samples:", X_train.shape)

# SVM Model with Grid srch
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(), svm_params, cv=3)
svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Best SVM Parameters: {svm_grid.best_params_}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Random Forest Model with Grid srch
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Best Random Forest Parameters: {rf_grid.best_params_}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# MLP Model with with Grid srch
mlp_params = {
    'hidden_layer_sizes': [(64, 32), (128, 64)],
    'max_iter': [300, 500]
}
mlp_grid = GridSearchCV(MLPClassifier(random_state=42), mlp_params, cv=3)
mlp_grid.fit(X_train, y_train)

best_mlp = mlp_grid.best_estimator_
y_pred_mlp = best_mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"Best MLP Parameters: {mlp_grid.best_params_}")
print(f"MLP Accuracy: {mlp_accuracy:.4f}")

