Plant Disease Detection

Introduction

This project aims to develop a system for detecting plant diseases using image processing and machine learning techniques. By identifying plant diseases early, this system can help improve crop health, reduce pesticide usage, and increase agricultural productivity.

Features

Automated Disease Detection: Identify plant diseases from images of leaves.

Early Intervention: Provides early warnings to prevent disease spread.

High Accuracy: Uses advanced machine learning models for reliable results.

Cost-Effective: Reduces reliance on manual inspections and unnecessary pesticide usage.

Technologies Used

Programming Language: Python

Libraries:

OpenCV (Image Processing)

TensorFlow/Keras (Deep Learning)

NumPy (Numerical Computations)

Scikit-learn (Data Splitting and Evaluation)

Setup Instructions

Step 1: Install Dependencies

Ensure Python is installed (version 3.8 or higher).
Run the following command to install the required libraries:
pip install tensorflow opencv-python numpy scikit-learn
Step 2: Prepare the Dataset

Organize the dataset as follows:
  dataset/
├── healthy/
│   ├── image1.jpg
│   ├── image2.jpg
├── disease1/
│   ├── image1.jpg
│   ├── image2.jpg
└── disease2/
    ├── image1.jpg
    ├── image2.jpg

    Each subfolder represents a class (e.g., Healthy, Disease1, Disease2).

Update the data_dir variable in the code with the dataset's path.

Step 3: Save the Code

Save the provided Python code in a file named plant_disease_detection.py.

Step 4: Run the Code

Run the script in your terminal or command prompt:
  python plant_disease_detection.py

  Step 5: Test the Model

Provide a test image path in the test_image_path variable to classify a sample image. The model will output the predicted class (e.g., "Healthy" or "Diseas

Project Workflow

Data Preprocessing:

Images are resized to a uniform size (128x128 pixels).

Pixel values are normalized to scale between 0 and 1.

Data augmentation (e.g., flipping, rotation) is applied to expand the dataset.

Model Building:

A Convolutional Neural Network (CNN) is used for image classification.

The architecture includes convolutional, pooling, and fully connected layers.

Model Training:

Dataset is split into training and validation sets.

Model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.

Testing:

Model performance is evaluated using unseen test images.

Accuracy and predictions are reported.

