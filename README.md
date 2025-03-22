# VR_Mini_Project_1

(a) Binary Classification Using Handcrafted Features and ML Classifiers
Feature Extraction
Images were converted to grayscale and resized to 64x64 for uniformity.

Histogram of Oriented Gradients (HOG) was used as the primary feature extraction method.

Dataset Preparation
The dataset consists of two categories: "with mask" and "without mask."

Features were extracted from both categories and normalized using StandardScaler.

The data was split into training (80%) and testing (20%) sets.

Machine Learning Classifiers
Three different classifiers were trained and evaluated:

Support Vector Machine (SVM)

Used with hyperparameter tuning via GridSearchCV.

Achieved an accuracy of 91.7%.

Random Forest Classifier

Optimized using GridSearchCV.

Achieved an accuracy of 87.3.

Multi-Layer Perceptron (MLP) Neural Network

Tuned for hidden layer sizes and iterations.

Achieved an accuracy of 91.3.

Observations
SVM provided strong results with optimized hyperparameters.

Random Forest performed well, offering a good balance between accuracy and interpretability.

The MLP classifier showed competitive performance, demonstrating the potential of neural networks even with handcrafted features.
