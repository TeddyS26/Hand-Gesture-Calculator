# Hand-Gesture-Calculator

## Project Overview

The Hand-Gesture Calculator is a real-time application that detects numerical hand gestures (0-9) using a webcam and performs basic arithmetic operations (addition, subtraction, multiplication, and division) based on those gestures. The user can select an operation using on-screen buttons, and the application calculates the result of the two recognized gestures.

### Features:

- Gesture Recognition: Detects hand gestures (0-9) in real-time using a trained Convolutional Neural Network (CNN).
- Basic Arithmetic Operations: Supports addition, subtraction, multiplication, and division, selected via on-screen buttons.
- Real-Time Prediction: Uses a webcam to capture hand gestures and displays the result of the arithmetic operation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [How to Run the Project](#how-to-run-the-project)
   - [Step 1: Clone the Repository](#step-1-clone-the-repository)
   - [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
   - [Step 3: Install Dependencies](#step-3-install-dependencies)
   - [Step 4: Run the Real-Time Gesture Recognition System](#step-4-run-the-real-time-gesture-recognition-system)
6. [Technical Details](#technical-details)
7. [Future Improvements](#future-improvements)

## How to Run the Project

### Step 1: Clone the Repository
Clone the repository to your local machine:
```
git clone https://github.com/TeddyS26/Hand-Gesture-Calculator.git
```

### Step 2: Create and Activate a Virtual Environment
Next you'll need to create a virtual environment:

#### 1. Create a Virtual Environment:

- For Windows:
   ```
   python -m venv hand-gesture-calculator-env
   ```

- For Mac/Linux:
   ```
   python3 -m venv hand-gesture-calculator-env
   ```

#### 2. Activate the Virtual Environment:

- For Windows:
   ```
   hand-gesture-calculator-env\Scripts\activate
   ```

- For Mac/Linux:
   ```
   source hand-gesture-calculator-env/bin/activate
   ```

### Step 3: Install Dependencies
Once the environment is active, install the necessary dependencies by running:

***Note: Python 3.7 or later is required to run the project.***
   ```
   pip install -r requirements.txt
   ```

### Step 4: Run the Real-Time Gesture Recognition System
To start the real-time gesture recognition system and perform arithmetic operations:
- For Windows:
   ```
   python main.py
   ```

- For Mac/Linux:
   ```
   python3 main.py
   ```

## Technical Details

- **Model Architecture**: The model is a feed-forward neural network trained on 3D hand landmarks (x, y, z) extracted using MediaPipe.
- **Libraries Used**:
  - **MediaPipe**: Detects and tracks 3D hand landmarks in real-time.
  - **TensorFlow/Keras**: Builds, trains, and saves the neural network model for gesture recognition.
  - **PyQt5**: Provides a user-friendly graphical interface for selecting operations and displaying results.
  - **OpenCV**: Captures the webcam feed, processes video frames, and displays the live camera view.

## Future Improvements

1. **Accuracy Improvement**:
   - Collect more diverse training data for gestures in different lighting and angles.
2. **Support for More Gestures**:
   - Extend the system to recognize additional hand gestures or complex sequences.