# Facial-Emotion-Recognition-with-CNNs-and-Transfer-Learning
Developed a facial emotion recognition system using deep learning and transfer learning. Benchmarked a custom CNN against ResNet50 and EfficientNet-B0 on the FER2013 dataset. The final EfficientNet-B0 model achieved 68.2% test accuracy and 67.0% macro F1-score, outperforming the custom baseline CNN by over 16 percentage points in accuracy.

# Facial Emotion Recognition with Deep CNNs and Transfer Learning: A Benchmark Study

## Overview

This project develops and evaluates deep learning models for **Facial Emotion Recognition (FER)** using the FER2013 dataset. The study compares a custom-built Convolutional Neural Network (CNN) against two state-of-the-art transfer learning architectures, ResNet50 and EfficientNet-B0, to determine the most effective approach for multi-class emotion classification.

The project follows a complete machine learning workflow including:

* Exploratory Data Analysis (EDA)
* Data preprocessing and augmentation
* Baseline CNN development
* Transfer learning with pretrained architectures
* Hyperparameter optimization
* Model comparison and benchmarking
* MLflow experiment tracking
* Grad-CAM visual explanations
* Real-time webcam emotion prediction
* Model export for deployment

---

## Objective

The goal of this project is to investigate whether transfer learning significantly improves facial emotion recognition performance compared to a custom CNN trained from scratch.

The models classify facial expressions into seven emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## Dataset

**FER2013**

* 35,887 grayscale facial images
* Resolution: 48 × 48 pixels
* 7 emotion classes
* Public benchmark dataset for emotion recognition

---

## Project Pipeline

### 1. Exploratory Data Analysis

Performed:

* Class distribution analysis
* Pixel intensity analysis
* Sample image visualization
* Emotion frequency comparisons

---

### 2. Data Preprocessing

Techniques applied:

* Image resizing
* Normalization using ImageNet statistics
* Data augmentation:

  * Random horizontal flips
  * Random rotations
  * Random affine transformations

---

### 3. Baseline CNN

A custom CNN was developed as a benchmark model.

Architecture:

* 4 Convolutional Blocks
* Batch Normalization
* Max Pooling
* ReLU Activation
* Dropout Regularization
* Fully Connected Classification Head

Loss Function:

* Cross Entropy Loss

Optimizer:

* AdamW

Learning Rate Scheduling:

* ReduceLROnPlateau

Early Stopping:

* Validation Accuracy Monitoring

---

### 4. Transfer Learning Models

#### ResNet50

Pretrained on ImageNet.

Training Strategy:

1. Freeze backbone
2. Train classification head
3. Unfreeze all layers
4. Fine-tune entire network

---

#### EfficientNet-B0

Pretrained on ImageNet.

Training Strategy:

1. Freeze backbone
2. Train classification head
3. Unfreeze all layers
4. Fine-tune entire network

---

### 5. Experiment Tracking

Experiments were tracked using MLflow.

Logged Information:

* Training loss
* Validation loss
* Accuracy
* Precision
* Recall
* F1 Score
* Hyperparameters
* Best model checkpoints

---

### 6. Explainable AI

Grad-CAM visualizations were generated to highlight facial regions influencing model predictions.

This improves interpretability and helps validate model decision-making.

---

### 7. Real-Time Inference

A webcam-based inference pipeline was developed using OpenCV.

Features:

* Face detection
* Real-time emotion prediction
* Probability visualization
* Live video stream inference

---

## Model Comparison

| Model           | Accuracy | Precision | Recall | F1 Score |
| --------------- | -------- | --------- | ------ | -------- |
| Baseline CNN    | 0.5187   | 0.4644    | 0.4548 | 0.4531   |
| ResNet50        | 0.6816   | 0.6839    | 0.6617 | 0.6709   |
| EfficientNet-B0 | 0.6882   | 0.6879    | 0.6685 | 0.6766   |

---

## Final Test Results

### Best Model: EfficientNet-B0

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.6817 |
| Precision | 0.6722 |
| Recall    | 0.6688 |
| F1 Score  | 0.6702 |

---

## Per-Class Performance

| Emotion  | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| Angry    | 0.5881    | 0.6169 | 0.6021   |
| Disgust  | 0.7184    | 0.6667 | 0.6916   |
| Fear     | 0.5371    | 0.5234 | 0.5302   |
| Happy    | 0.8709    | 0.8743 | 0.8726   |
| Neutral  | 0.6330    | 0.6350 | 0.6340   |
| Sad      | 0.5644    | 0.5373 | 0.5505   |
| Surprise | 0.7935    | 0.8279 | 0.8104   |

---

## Key Findings

* Transfer learning dramatically outperformed the custom CNN.
* EfficientNet-B0 achieved the best overall performance.
* The baseline CNN achieved 51.9% test accuracy.
* EfficientNet-B0 improved performance to 68.2% test accuracy.
* Happy and Surprise emotions were classified most accurately.
* Fear and Sad remained the most challenging classes.
* Grad-CAM visualizations confirmed that the models focused on meaningful facial regions.

---

## Technologies Used

### Deep Learning

* PyTorch
* Torchvision

### Machine Learning

* Scikit-learn
* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Experiment Tracking

* MLflow

### Computer Vision

* OpenCV

### Explainability

* Grad-CAM

---

## Repository Structure

```
├── notebooks/
│   └── facial_emotion_recognition.ipynb
│
├── models/
│   ├── baseline_cnn.pth
│   ├── resnet50_best.pth
│   └── efficientnet_b0_best.pth
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── gradcam_examples.png
│   └── training_curves.png
│
├── mlruns/
│
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Vision Transformers (ViT)
* ConvNeXt architectures
* Hyperparameter optimization with Optuna
* Ensemble methods
* Model deployment with FastAPI
* ONNX optimization for edge devices
* Real-time web application deployment

---

## Author

Developed as a deep learning portfolio project demonstrating:

* Computer Vision
* Deep Learning
* Transfer Learning
* Model Evaluation
* Explainable AI
* MLOps Experiment Tracking
* Real-Time Inference Systems
