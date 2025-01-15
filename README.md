# Deep-Learning-Lung-Sound-Analysis
This project involves the development of a deep learning-based lung sound analysis system for an intelligent stethoscope. The system is designed to analyze respiratory sounds to detect abnormal lung conditions such as asthma, pneumonia, and COPD.
 # Deep Learning-Based Lung Sound Analysis System

## Overview
This project focuses on developing a **deep learning-based lung sound analysis system** for intelligent stethoscopes. The system is designed to analyze respiratory sounds and detect abnormal lung conditions such as asthma, pneumonia, and COPD. It integrates advanced signal processing techniques and machine learning to provide instant diagnostic feedback, offering portability and usability in remote or resource-limited settings.

---

## Features
- **Signal Acquisition**: Captures lung sounds using a high-sensitivity microphone integrated into the stethoscope.
- **Preprocessing**: Filters recordings using Advanced Digital Signal Processing (ADSP) methods to remove noise and highlight key features.
- **Feature Extraction**: Extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** for model training.
- **Deep Learning Models**:
  - Fully Connected Neural Network (FNN)
  - Convolutional Neural Network (CNN)
- **Classification**: Detects and classifies respiratory conditions.
- **Portability**: Designed for easy use in remote or resource-limited settings.
- **IoT Integration**: Enables remote monitoring and diagnosis.

---

## Project Structure

Project Work/ ├── src/ │ ├── Classifier_train_and_test.py # Main script for training and testing │ ├── Deep_learning.py # Core deep learning models and utilities │ ├── Evaluation_metrics.py # Performance metric computations │ └── config.py # Configuration and path settings ├── data/ │ ├── train/ # Training data │ ├── test/ # Testing data │ └── processed/ # Preprocessed data files ├── results/ # Model outputs and logs └── README.md # Project documentation


---

## Installation

### Requirements
- Python 3.8 or higher
- Required Libraries:
  ```bash
  pip install -r requirements.txt
