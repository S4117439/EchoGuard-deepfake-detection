# EchoGuard - Deepfake Audio Detection System

## Overview
EchoGuard is a machine learning-based system designed to detect AI-generated (deepfake) audio using acoustic feature extraction and classification.  
It allows users to upload audio files and receive a prediction (real or AI-generated), along with a confidence score and explanation.

---

## Features
- Upload audio files (WAV, MP3, FLAC, M4A, etc.)
- Detect whether audio is real or AI-generated
- Confidence score output
- Explanation based on extracted features
- Web interface (HTML/CSS/JavaScript)
- Flask backend API

---

## Technologies Used
- Python
- Flask
- Librosa
- Scikit-learn (Random Forest)
- HTML/CSS/JavaScript

---

## Project Structure

## Screenshots

### Real Audio Detection
The system correctly classified a real human speech sample as authentic.

![Real Audio Result](screenshots/real_results.png)

### Fake Audio Detection
The system correctly classified an AI-generated speech sample as synthetic.

![Fake Audio Result](screenshots/fake_results.png)