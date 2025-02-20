# Handwritten-Digit-Recognition
A deep learning model (CNN) that classifies handwritten digits.

Objective:
This project builds an AI-powered handwritten digit recognition system using Convolutional Neural Networks (CNNs). It can classify handwritten digits (0-9) by processing an image containing a digit.

How It Works:

User Uploads an Image – The user uploads an image containing a handwritten digit.
Preprocessing – The image is grayscale-converted, resized (28x28 pixels), and normalized.
CNN Model Prediction – The pre-trained CNN model (trained on MNIST dataset) predicts the digit.
Result Display – The predicted digit is displayed on the webpage.

Tech Stack:

Backend: Flask, TensorFlow/Keras, OpenCV
Frontend: HTML, CSS, JavaScript
ML Algorithm: CNN (trained on MNIST dataset)
Dataset Used: MNIST Handwritten Digit Dataset

Theory Behind Handwritten Digit Recognition
CNNs (Convolutional Neural Networks) are used for image recognition tasks. They extract important features from images using multiple layers, such as:

Key CNN Layers Used:
1)Convolutional Layer – Detects edges, shapes, and patterns.
2)Pooling Layer – Reduces image size while preserving features.
3)Fully Connected Layer – Classifies the image into categories (digits 0-9).

The MNIST dataset (70,000 images of handwritten digits) is used to train the CNN model.
