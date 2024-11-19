# Hand-Gesture-Calculator

## Project Overview

The Hand-Gesture Calculator is a real-time application that detects numerical hand gestures (0-9) using a webcam and performs basic arithmetic operations (addition, subtraction, multiplication, and division) based on those gestures. The user can select an operation using on-screen buttons, and the application calculates the result of the two recognized gestures.

### Features:

- Gesture Recognition: Detects hand gestures (0-9) in real-time using a trained Convolutional Neural Network (CNN).
- Basic Arithmetic Operations: Supports addition, subtraction, multiplication, and division, selected via on-screen buttons.
- Real-Time Prediction: Uses a webcam to capture hand gestures and displays the result of the arithmetic operation.


## How to Run the Project

### Step 1: Clone the Repository
Clone the repository to your local machine:
```
git clone https://github.com/TeddyS26/Hand-Gesture-Calculator.git
```

### Step 2: Create and Activate a Virtual Environment
Next you'll need to create a virtual enviorment:

1. Create a virtual environment:
```
python -m venv hand-gesture-calculator-env
```

2. Activate the virtual environment:
```
hand-gesture-calculator-env\Scripts\activate
```

### Step 3: Install Dependencies
Once the environment is active, install the necessary dependencies by running:
```
pip install -r requirements.txt
```

### Step 4: Run the Real-Time Gesture Recognition System
To start the real-time gesture recognition system and perform arithmetic operations:
```
python main.py
```