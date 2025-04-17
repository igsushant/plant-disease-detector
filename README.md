# plant-disease-detector
This project focuses on building an AI-powered system to detect plant diseases from leaf images using deep learning and deploying it as an interactive web application using Streamlit.

Objective
To help farmers and agricultural experts identify plant diseases early by analyzing leaf images with a simple camera interface or file upload — reducing crop loss and enabling timely treatment.

Workflow Overview
Dataset Used

PlantVillage Dataset from Kaggle

~54,000 labeled images of healthy and diseased leaves

38 classes across multiple plant species (e.g., tomato, apple, corn)

Model Development

Framework: TensorFlow and Keras

Model: Transfer Learning using MobileNetV2

Preprocessing: Image resizing, normalization, data augmentation

Trained on ~80% of data, validated on 20%

Final model saved as: plant_disease_model.h5

Classes stored in classes.txt for use during prediction

Web App Development

Framework: Streamlit

Features:

Upload an image or capture with live camera

Displays prediction result with confidence score

Option to reset and re-upload

Backend loads the saved .h5 model and predicts the class

Deployment

Final .h5 model and app deployed on Streamlit Cloud

Hosted as a free, online application accessible from any browser

No Python installation required — shareable via link

Technologies Used
Python, TensorFlow, Keras

MobileNetV2 (pretrained model)

Streamlit (UI & deployment)

PIL, NumPy (image processing)

PyInstaller (for local executable version)

Outcome
A working AI web app for plant disease detection

Can be used by farmers, students, or researchers

Easy to use, mobile-friendly, and publicly accessible online
