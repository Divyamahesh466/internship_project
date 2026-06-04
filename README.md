# Face Tracking Attendance System

## Project Overview

The Face Tracking Attendance System is a Computer Vision project that detects and recognizes individuals from a video, monitors their movement within a predefined boundary region, and automatically logs entry and exit events.

The system combines face detection, face recognition, boundary monitoring, state management, and CSV-based attendance logging into a single integrated pipeline.

---

## Features

* Face Detection using OpenCV
* Face Recognition using LBPH Face Recognizer
* Boundary (Geofence) Monitoring
* State Management (OUTSIDE → ENTERED → INSIDE → EXITED)
* CSV Attendance Logging
* Real-Time Visual Feedback
* Video-Based Processing

---

## System Workflow

Video Input
↓
Face Detection
↓
Face Recognition
↓
Boundary Check
↓
State Management
↓
CSV Log Generation

---

## Project Structure

Face_Tracking_System/

├── models/
│   ├── trainer.yml
│   └── labels.txt
│
├── config.py
├── logger.py
├── state_manager.py
├── train_model.py
├── integrated_demo.py
├── requirements.txt
├── README.md
└── .gitignore

---

## State Machine

OUTSIDE
↓
ENTERED
↓
INSIDE
↓
EXITED
↓
OUTSIDE

---

## Attendance Log Format

Timestamp,Person_Name,State

2026-06-04 15:49:21,Ganesh,ENTERED

2026-06-04 15:50:03,Ganesh,EXITED

---

## Installation

Clone the repository:

git clone https://github.com/babu2004/face_tracking_system.git

Navigate to the project directory:

cd face_tracking_system

Install dependencies:

pip install -r requirements.txt

---

## Dataset Setup

Create a dataset folder with one folder per person:

dataset/

├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│
├── Person2/
│   ├── image1.jpg
│   ├── image2.jpg

---

## Training

Train the recognition model:

python train_model.py

This generates:

* trainer.yml
* labels.txt

inside the models folder.

---

## Running the System

Place the input video inside the videos folder and update the path in config.py.

Run:

python integrated_demo.py

The system will:

* Detect faces
* Recognize people
* Monitor zone entry/exit
* Update state transitions
* Generate attendance logs

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Pillow
* LBPH Face Recognition

---

## Future Improvements

* YOLO-based face/person detection
* DeepFace / FaceNet recognition
* Multi-person tracking
* Real-time webcam support
* Database integration
* Dashboard and analytics
* Cloud deployment

---

## Team Members

* Ganesh Babu
* Team Member 2
* Team Member 3
* Team Member 4
* Team Member 5
* Team Member 6
* Team Member 7

---

## Sprint 1 Deliverables

* Dataset Preparation
* Face Detection Module
* Face Recognition Module
* Boundary Monitoring
* State Management
* CSV Logging
* End-to-End Integration
