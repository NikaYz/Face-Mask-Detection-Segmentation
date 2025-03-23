# VR_Mini_Project_1

<!-- (a) Binary Classification Using Handcrafted Features and ML Classifiers
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

Observations:
SVM provided strong results with optimized hyperparameters.

Random Forest performed well, offering a good balance between accuracy and interpretability.

The MLP classifier showed competitive performance, demonstrating the potential of neural networks even with handcrafted features. -->
# Face Mask Classification and Segmentation

## Introduction

### Objective  
Develop a computer vision solution to classify and segment face masks in images. The project involves using handcrafted features with machine learning classifiers and deep learning techniques to perform classification and segmentation.

## Dataset  

### Dataset Source  
A labeled dataset containing images of people with and without face masks can be accessed here: [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset).  

### Dataset Structure  
It has two sections:  
- **Masked Faces**  
- **Unmasked Faces**

## Steps taken to achieve each task

### Feature Extraction


### Model training
#### CNN
Built a **CNN** with **two convolutional layers**, followed by **ReLU activations**, **max pooling**, and **fully connected layers** for classification.  

Used **CrossEntropyLoss** as the loss function and **Adam optimizer** with a learning rate of **0.001**.  
Trained for **10 epochs**, updating weights after each batch to minimize loss.  


### Segmentation Techniques

 
## Hyperparameter tuning

### CNN
- **Learning rate:** Tested **0.001, 0.0005, 0.0001**, with **0.001** performing the best.  
- **Number of filters:** Tried **32, 64, 128** for convolutional layers, with **(32 → 64)** providing a good balance.  
- **Kernel sizes:** Compared **3×3** and **5×5**, and found **3×3** to be more effective.  
- **Fully connected layer sizes:** Tested **128, 256, 512**, and **128** performed well without overfitting. 
### UNet Model


## Results

### Binary Classification 
The models were evaluated on the test set, and their classification accuracies were as follows:

- **SVM Accuracy:** **91.45%**  
- **Random Forest Accuracy:** **88.89%**  
- **CNN Accuracy:** **95.73%**  
- **MLP Accuracy:** **80.95%**


### Segmentation


## Observations And Final Analysis
 ### Binary Classification
 As it turns out CNN outperformed traditional machine learning classification models, however it was computationally more challenging to train as compared to other models.Moreover there were several hyperparameters to fine tune the model as compared to other algorithms.

 ### Segmentation


## How to Run Code
