# Motorcycle Brand Prediction

## Overview

This project implements a deep learning solution to classify motorcycle images by brand. The model can identify various motorcycle manufacturers such as BMW, Honda, Kawasaki, Ducati, and others from images. It leverages transfer learning and progressive training techniques to achieve high accuracy.

---

## Features

- Image classification across 19 motorcycle brands
- Built on ResNet50 architecture with transfer learning
- Progressive unfreezing training approach for optimized accuracy
- DirectML GPU acceleration for training performance
- Data augmentation to improve model generalization
- Class-weighted loss to handle dataset imbalance

---

## Technologies

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.x, PyTorch |
| **Acceleration** | torch_directml |
| **Image Processing** | torchvision, PIL |
| **Data Analysis** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |

---

## Project Structure

```
motorcycle-brand-prediction/
├── data/                      # Training/validation/testing data splits
├── cache/                     # Cache files for faster dataset loading
├── checkpoints/               # Model checkpoints saved during training
├── images/                    # Source images and processing scripts
├── processed_data/            # Processed data files and metadata
├── motorcycle-classifier.ipynb # Main training notebook
└── best_resnet_motorcycle_classifier.pth  # Best trained model weights
```

---

## Model Architecture

The classification model uses a ResNet50 backbone pretrained on ImageNet with a custom classifier head:

```
ResNet50 Backbone
    ↓
2048-unit FC Layer → BatchNorm → ReLU → Dropout(0.3)
    ↓
1024-unit FC Layer → BatchNorm → ReLU → Dropout(0.3)
    ↓
Output Layer (num_classes)
```

---

## Training Approach

1. **Data Preparation**
   - Dataset splitting with stratified sampling
   - Class balancing to handle imbalanced distributions

2. **Model Training**
   - Progressive unfreezing of ResNet layers
   - Gradient accumulation for effective larger batch sizes
   - Early stopping to prevent overfitting
   - Learning rate scheduling with OneCycleLR

3. **Data Augmentation**
   - Random resized crops and flips
   - Color jittering and random erasing
   - Perspective and affine transformations

---

## Usage

1. **Dataset Preparation**
   - Organize motorcycle images by brand
   - Run data preparation scripts to create training splits

2. **Model Training**
   - Execute the training notebook with desired parameters
   - Monitor training progress with validation metrics

3. **Evaluation**
   - Test the model on held-out test images
   - Analyze confusion matrix and classification report

4. **Inference**
   - Use the trained model to predict brands of new motorcycle images

---

*Note: This project is for educational purposes only.*