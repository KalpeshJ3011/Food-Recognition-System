# Food Recognition System Using Deep Learning

## Overview

The Food Recognition System is a Deep Learning based application that identifies Indian food items from images.

The project uses MobileNetV2 with Transfer Learning to classify food images efficiently while maintaining high accuracy and low computational cost.

## Features

- Food image classification
- MobileNetV2 Transfer Learning
- TensorFlow & Keras implementation
- Data Augmentation
- Real-time prediction
- TensorFlow Lite deployment
- Gradio Web Interface

## Dataset

Indian Food Classification Dataset from Kaggle:

https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Gradio

## Project Structure

Food-Recognition-System/

├── README.md

├── requirements.txt

├── LICENSE

├── dataset/

├── models/

├── notebooks/

├── src/

├── screenshots/

└── docs/

## Model Architecture

- MobileNetV2 (Pre-trained on ImageNet)
- Global Average Pooling Layer
- Dropout Layer
- Dense Layer (ReLU)
- Softmax Output Layer

## Training Configuration

- Image Size: 224 × 224
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Categorical Crossentropy
- Validation Split: 20%
- Epochs: 10

## Results

The model successfully classifies multiple Indian food categories with high accuracy.

Examples of classes:

- Burger
- Butter Naan
- Chai
- Chapati
- Chole Bhature
- Dal Makhani
- Dhokla
- Fried Rice
- Idli
- Jalebi
- Kadai Paneer
- Masala Dosa
- Momos
- Pani Puri
- Pav Bhaji
- Pizza
- Samosa

## Applications

- Diet Tracking
- Nutrition Monitoring
- Restaurant Automation
- Smart Food Ordering Systems
- Calorie Estimation Systems

## Future Improvements

- Mobile Application Deployment
- Calorie Estimation
- Nutrition Analysis
- Larger Dataset Support
- Real-Time Camera Detection

## Author

Final Year Computer Engineering Project

Konkan Gyanpeeth College of Engineering, Karjat

2025-2026
