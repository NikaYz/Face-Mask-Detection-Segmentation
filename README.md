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
## Repo Structure  

- **Binary_Classification.ipynb**  
  - Contains code for binary classification of face mask detection using **ML classifiers** and **CNN models**. 
- **RegionSegmentationUsingThresholding**  
  - Includes code for traditional segmentation approaches for face mask segmentation.
- **Mask_Segmentation_Using_U-Net/** 
  - Contains code for U-Net model used for face mask segmentation. 
- **VR Mini Project 1.pdf**
  - Provides a detailed problem description and the source of the dataset.  
- **images/**  
  - Stores images used in the README.md for reference.  
- **dataset/**  
  - Contains the dataset required for binary classification of face masks. 

## Dataset  

### Dataset Source  
#### For Binary Classification
A labeled dataset containing images of people with and without face masks can be accessed here: [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset).  
#### For Region and Mask Segmentation  
A labeled dataset containing cropped face images along with their ground truth masks can be accessed in the download section here: [MSFD (Masked Face Segmentation Dataset)](https://github.com/sadjadrz/MFSD)

### Dataset Structure 
#### For Binary Classification
It has two sections:  
- **Masked Faces**  
- **Unmasked Faces**
  
#### For  Region and Mask Segmentation  
The dataset is organized into the following structure:  
- **1/** → Main folder containing all dataset files  
  - **img/** → Contains original images from which faces were cropped  
  - **dataset.csv** → Provides bounding box sizes of detected faces  
  - **face_crop/** → Stores all cropped face images  
  - **face_crop_segmentation/** → Contains segmented images of cropped faces  
    - Segmented images have the same filename as their corresponding cropped face images
    - Note: One file in this folder has a mismatched filename, which caused an issue during training but was later fixed in the code. 

## Steps taken to achieve each task

### Feature Extraction
To extract meaningful features from the dataset, we used Histogram of Oriented Gradients (HOG). 

- **Grayscale Conversion:** Images were first converted to grayscale to reduce complexity and focus on structural features rather than color.
- **Resizing:** Images were resized to **64x64 pixels** to standardize the input dimensions for the model.
- **HOG Features:** The HOG method was applied to each image to capture gradient information, which is useful for edge detection and texture patterns. The parameters used were:
  - **Pixels per cell:** (8, 8)
  - **Cells per block:** (2, 2)
  - **Feature vector:** True (flatten the HOG features into a single vector).

This resulted in a vector of features that was then fed into machine learning classifiers.

### Model training

#### SVM (Support Vector Machine)
- Used with hyperparameter tuning via GridSearchCV.

#### Random Forest Classifier
- Optimized using GridSearchCV.

#### Multi-Layer Perceptron (MLP) Neural Network
- Tuned for hidden layer sizes and iterations.

#### CNN
Built a **CNN** with **two convolutional layers**, followed by **ReLU activations**, **max pooling**, and **fully connected layers** for classification.  

Used **CrossEntropyLoss** as the loss function and **Adam optimizer** with a learning rate of **0.001**.  
Trained for **10 epochs**, updating weights after each batch to minimize loss.  


### Segmentation Techniques


- Normal Thresholding
- Otsu’s Thresholding
- K-Means Clustering: Seed points chosen randomly or from lower-middle part of image
- Region Growing: Seed points chosen randomly or from lower-middle part of image
- Custom Approach: Masked upper part of the image, applied inverse and binary thresholding

### U-Net Model

- Dataset Size: Initial tuning on 4,000 images, full training in three batches
- Memory Constraints: Managed by splitting dataset into batches
- Issue Encountered: Mismatched file caused input-output misalignment, later corrected


 
## Hyperparameter tuning

#### SVM
The Support Vector Machine (SVM) classifier was fine-tuned using **GridSearchCV**:
- **C values:** [0.1, 1, 10] for regularization strength.
- **Kernel types:** ['linear', 'rbf']

#### Random Forest
GridSearchCV was used to tune the Random Forest classifier:
- **Number of estimators:** [50, 100, 200]
- **Max depth:** [10, 20, None] to control tree depth.
- **Min samples split:** [2, 5, 10]

#### MLP (Multi-Layer Perceptron)
We optimized the MLP model by testing different combinations of hidden layer sizes:
- **Hidden Layer Sizes:** [(64, 32), (128, 64)].
- **Max iterations:** [300, 500].

### CNN
- **Learning rate:** Tested **0.001, 0.0005, 0.0001**, with **0.001** performing the best.  
- **Number of filters:** Tried **32, 64, 128** for convolutional layers, with **(32 → 64)** providing a good balance.  
- **Kernel sizes:** Compared **3×3** and **5×5**, and found **3×3** to be more effective.  
- **Fully connected layer sizes:** Tested **128, 256, 512**, and **128** performed well without overfitting.
- 
### U-Net Model
- **Learning Rate:** Tried **0.001, 0.0005, 0.0001**, → Best: **0.001**
- **Batch Size:** Tried **8, 16, 32**, → Best: **8** as provides stability over training
- **Epochs:** Tried **30, 50, 100**, → Used: **30** (Computational constraints)
- **Loss Function:** Tried **Binary cross-entropy, Dice score** → Best: **Binary cross-entropy**
- **Dropout Rate:** With/without dropout → Best: **0.1**
- **Image Size:** Tried **128×128, 256×256** → Used: **128×128** (Computational constraints)
- **Data Augmentation:** Applied  


## Results

### Binary Classification 
The models were evaluated on the test set, and their classification accuracies were as follows:

- **SVM Accuracy:** **91.45%**  
- **Random Forest Accuracy:** **88.89%**  
- **CNN Accuracy:** **95.73%**  
- **MLP Accuracy:** **80.95%**

### Segmentation  
#### Traditional Segmentation for a random image visualization:
<div style="text-align: center;">
        <img src="images/Traditional_segmentation.png" alt="Visualization" width="600" height="400">
        <p>Different traditional segmentation evaluation</p>
 </div>

#### U-Net Model Performance on Test Data  
- **Accuracy:** **61.40%***  
- **Dice Coefficient:** **0.8635**  
- **IoU Metric:** **0.7599**  

#### Comparison with Traditional Segmentation for a random image  

 <div style="text-align: center;">
        <p>Comparision with Traditional Segmentation for random image with evaluation metric</p>
        <img src="images/face_segmentation_example.png" alt="Visualization" width="400" height="150">      
 </div>


## Observations And Final Analysis
 ### Binary Classification
 As it turns out CNN outperformed traditional machine learning classification models, however it was computationally more challenging to train as compared to other models.Moreover there were several hyperparameters to fine tune the model as compared to other algorithms.

### Segmentation  
As it turns out, U-Net outperformed traditional segmentation methods but was computationally intensive to train. Among traditional methods, Region Growing(with the seed point at the lower middle part of the image) and a custom segmentation approach (using inverse and binary thresholding) performed better than other methods.


## How to Run the Code  

This project consists of a **Python notebook**, which can be easily executed by downloading and running it.  

### For Segmentation:  
- The dataset must be uploaded to Kaggle.  
- A Kaggle API key is required to access the dataset within the notebook.  
- Ensure the API key is properly set up before running the segmentation code.

## Member-wise Contribution

- **Harsh Dhruv** (@hrdhruv)  
  - Implemented Binary Classification Using Handcrafted Features and ML Classifiers.
  - Assisted in training the Mask Segmentation model (collaborated with Aditya). 

- **Rudra Pathak** (@rudra0000)  
  - Implemented Binary Classification using CNN with comparsion with ML classifiers. 
  - Contributed to Region Segmentation using Traditional Technique with custom thresholding approach (collaborated with Aditya).

- **Aditya Saraf** (@NikaYz)  
  - Implemeted Region Segmentation using Traditional Techniques (with assistance from Rudra).
  - Implemented Mask Segmentation using U-Net (with assistance from Harsh).



