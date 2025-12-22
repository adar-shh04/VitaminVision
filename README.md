# Vitamin Vision

Vitamin Vision is a **proof-of-concept full-stack machine learning application** that explores the feasibility of detecting vitamin-related information from food images using deep learning and computer vision techniques.

This project was developed as part of a **SmartInternz Summer Project Internship** and focuses on end-to-end system design rather than production-grade medical accuracy.


## 🚀 Problem Statement

Tracking vitamin intake is difficult for most individuals, and existing diet-tracking tools often focus on calories rather than micronutrients.  
Vitamin Vision explores whether **computer vision models** can be used to analyze food images and provide vitamin-related insights in a simplified, accessible way.


## 🧠 Project Overview

- Users upload an image of food through a web interface
- A deep learning model processes the image
- The system predicts vitamin-related categories based on learned visual patterns
- Results are displayed through a Flask-based web application

⚠️ Note:  
Vitamins are chemical properties and cannot be directly inferred from images.  
This project is intended as a **learning-oriented prototype**, not a medical or nutrition-grade system.


## 🛠️ Tech Stack

### Frontend
- HTML
- CSS

### Backend
- Python
- Flask

### Machine Learning
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)
- Transfer Learning (VGG19)

### Tools
- Git
- GitHub


## ⚙️ Model Details

- Compared **VGG16** and **VGG19** CNN architectures
- Used **VGG19 with ImageNet pre-trained weights**
- Applied data preprocessing and augmentation:
  - Image resizing
  - Normalization
  - Rotation and flipping
- Implemented regularization techniques and early stopping
- Achieved **moderate accuracy**, constrained by dataset quality and problem framing


## 🧩 System Architecture (High-Level)

1. User uploads food image
2. Image is preprocessed
3. CNN model performs classification
4. Prediction result is returned via Flask backend
5. Output is displayed on the web interface


## ▶️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ADARSH-SINGH-1/VitaminVision.git
   cd VitaminVision
   pip install -r requirements.txt
   python app.py
   http://127.0.0.1:5000

## Limitations & Learnings

- Vitamin content is not visually deterministic
- Dataset size and label quality limited model performance
- Similar-looking foods caused misclassification
-The project highlights the limitations of computer vision for nutritional inference

These challenges provided valuable insights into:

- ML problem framing
- Dataset bias
- Model evaluation beyond raw accuracy

