# Data Science Repository

Welcome to my Data Science repository! This repository contains homework assignments and a project developed as part of the **Sapienza's Fundamentals of Data Science** course. The assignments and project are focused on various data science techniques and methodologies, ranging from image processing to classification and advanced deep learning applications.

---

## Repository Structure

- **[HW1](#homework-1-image-filtering-and-object-identification):** Image Filtering and Object Identification  
- **[HW2](#homework-2-classification):** Logistic Regression, Multinomial Classification, and Convolutional Neural Networks  
- **[Project](#project-identifying-plant-disease-types-using-leaf-images):** Identifying Plant Disease Types Using Leaf Images  

---

## Homework 1: Image Filtering and Object Identification

**Objective:**  
Explore image processing techniques, including image filtering, edge detection, multi-scale image representations, and object identification.  

**Tasks:**  
- Apply 1D and 2D filters to images.  
- Perform edge detection using Prewitt and Canny operators.  
- Detect corners using the Harris Corner Detector.  
- Build Gaussian pyramids and analyze aliasing.  
- Identify objects using histograms and evaluate their performance.  

This assignment focuses on implementing image processing techniques and analyzing their performance for tasks like image retrieval and object identification.

---

## Homework 2: Classification  

**Objective:**  
Implement classification models for binary and multi-class problems and explore Convolutional Neural Networks (CNNs) to improve prediction accuracy.  

**Tasks:**  
- Implement Logistic Regression with Gradient Ascent.  
- Extend logistic regression with polynomial features and apply regularization.  
- Develop a multinomial classification pipeline using the Softmax Regression Model.  
- Build and evaluate a CNN model on the CIFAR-10 dataset.  
- Experiment with custom and pre-trained models to improve accuracy.  

This assignment emphasizes both traditional machine learning techniques and an introduction to deep learning concepts.

---

## Project: Identifying Plant Disease Types Using Leaf Images  

### Task Statement  
Predict the type of disease affecting a plant leaf from its image, with labels including "healthy," "rust," "scab," and "multiple diseases."  

### Motivation  
Early identification of plant diseases is crucial for reducing crop losses and improving food security. Automating this process eliminates the need for labor-intensive manual inspections.  

### Models  
- **Baseline:** Logistic Regression or Random Forest using basic image features like RGB histograms.  
- **Advanced:** Deep learning models, including CNN architectures such as ResNet or EfficientNet, for feature extraction and classification.  

### Tools and Libraries  
- Python  
- TensorFlow/Keras or PyTorch  
- Scikit-learn  

### Investigation Plan  
- Experiment with CNN architectures and apply data augmentation for better model generalization.  
- Use interpretability tools like SHAP or LIME to understand model predictions.  

### Dataset  
- **Source:** Kaggle Plant Pathology 2020 competition dataset (~3,000 labeled images).  
- **Preprocessing:** Normalize images and apply data augmentation techniques (e.g., rotations, flips).  

### Evaluation Metrics  
- Weighted F1-Score as the primary metric to address class imbalances.  
- Additional metrics: Accuracy, Precision, Recall, and Confusion Matrix.  

**Current Progress:**  
- A basic pipeline has been established with baseline models.  
- Initial results from CNN models are under review.  
