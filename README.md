# Saahaya – AI Powered Smart Surveillance & Human Detection System

## Project Overview

Saahaya is an AI-powered smart surveillance and human detection system developed using Python, Computer Vision, and Deep Learning technologies. The system analyzes surveillance videos in real-time and provides intelligent monitoring through human detection, face recognition, gender prediction, emotion analysis, and alert generation.

The project processes video footage using OpenCV and AI models to identify suspicious activities and improve smart security monitoring.

The main execution file of the project is:

```bash
s6.py
```

The sample surveillance video used for testing is:

```bash
walking.mp4
```

---

# Technologies Used

## Programming Language

- Python

## Libraries & Frameworks

- OpenCV
- NumPy
- Pandas
- PyTorch
- TensorFlow
- Keras
- Matplotlib

## AI & ML Technologies

- Computer Vision
- Deep Learning
- Face Detection
- Gender Classification
- Emotion Recognition

---

# Project Features

## Core Features

- Human detection system
- Face detection and recognition
- Gender prediction
- Emotion recognition
- Person counting
- Real-time video analytics
- Smart surveillance monitoring
- Alert generation system
- AI-based video processing
- Deep learning integration

## AI Features

- Real-time surveillance analysis
- AI-powered suspicious activity monitoring
- Video-based intelligent detection
- Emotion and gender analysis from video streams

---

# Project Structure

```bash
Saahaya/
│
├── s6.py
├── walking.mp4
├── gender_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Requirements File

Create a file named:

```bash
requirements.txt
```

Add the following content:

```txt
opencv-python
numpy
pandas
torch
torchvision
tensorflow
keras
matplotlib
scikit-learn
Pillow
```

---

# How to Run the Project

## Step 1 – Clone Repository

```bash
git clone https://github.com/your-username/Saahaya.git
cd Saahaya
```

---

## Step 2 – Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 3 – Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Step 4 – Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5 – Run the Project

```bash
python s6.py
```

---

# Input Video

The project processes:

```bash
walking.mp4
```

You can replace it with your own CCTV or surveillance footage.

---

# Output Generated

The system generates processed videos with:

- Human Detection
- Face Recognition
- Gender Prediction
- Emotion Analysis
- Smart Alerts

---

# AI Models Used

## Deep Learning Models

- Gender Detection Model
- Face Detection Algorithms
- Emotion Recognition Model

The trained AI model file used:

```bash
gender_model.pth
```

---

# Future Enhancements

- Weapon detection
- Crowd density analysis
- Cloud deployment
- Multi-camera support

---
