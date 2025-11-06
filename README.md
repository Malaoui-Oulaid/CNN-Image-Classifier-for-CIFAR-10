#  **CNN Image Classifier for CIFAR-10**

Welcome to my project on image classification! 🚀 This repository contains a Convolutional Neural Network (CNN) built from scratch using TensorFlow and Keras to classify images from the well-known CIFAR-10 dataset. The model is also capable of predicting labels for custom, real-world images.

---

##  Project Overview

### Objective
The main goal of this project is to build, train, and evaluate a robust CNN model for multi-class image classification. The project demonstrates a complete machine learning pipeline, from data loading and preprocessing to model training, evaluation, and practical inference on new images.

### Dataset
- **CIFAR-10**: A benchmark dataset consisting of 60,000 32x32 color images in 10 different classes.
  - **Classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

### Methodology
1.  **Data Loading & Preprocessing**:
    -   Loaded the CIFAR-10 dataset directly from `keras.datasets`.
    -   Normalized pixel values to a range of `[0, 1]` for better model performance.
2.  **Model Architecture**:
    -   A sequential CNN was built using Keras.
    -   The model consists of three convolutional blocks, each containing two Conv2D layers with ReLU activation and BatchNormalization, followed by a MaxPooling2D layer to reduce spatial dimensions and a Dropout layer to prevent overfitting.
    -  The number of filters increases progressively from 64 → 128 → 256 across the three blocks, allowing the network to learn increasingly complex features.
    -   After the convolutional blocks, a Flatten layer converts the 2D feature maps into a 1D feature vector.
    -   This is followed by a fully connected layer (Dense(512)) with ReLU activation, BatchNormalization, and Dropout(0.5) for further regularization.
    -   The final output layer is a Dense(10) layer with softmax activation, producing probability distributions over 10 target classes.
3.  **Training & Evaluation**:
    -   The model was compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss function.
    -   Trained for 5 epochs, achieving a validation accuracy of approximately **81%**.
4.  **Inference and Real-World Testing**:
    -   The trained model was used to make predictions on the test set.
    -   More importantly, it was tested on **custom images** (e.g., a airplan, a automobile) to prove its practical applicability.

---

##  Key Features

-   Implementation of a CNN from the ground up.
-   Complete training and validation pipeline.
-   Functionality to predict classes for any user-provided image.

---

##  Technologies Used

-   **Framework**: TensorFlow, Keras
-   **Libraries**: NumPy, Matplotlib, OpenCV (for image processing)

---

##  Results

The model was trained for 5 epochs, and the final performance on the validation set was:
-   **Validation Loss**: ~0.5872
-   **Validation Accuracy**: ~81.34%

The model demonstrated its effectiveness by correctly classifying custom images not present in the original dataset.


---

```

---
