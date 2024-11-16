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
python real_time_recognition.py
```

## Project Progress
### Week 1:
- Project Planning: Defined the project goals and requirements for the Hand Gesture Calculator.
- Environment Setup: Installed necessary tools, including Python, OpenCV, TensorFlow, and Keras.
- Initial File Structure: Set up the basic project structure and Git version control.

### Week 2:
- Data Collection (Part 1): Started collecting hand gesture images (0-9) using the webcam.
- image_capture.py: Created the script to capture and save images in gesture-specific directories.
- Data Organization: Set up directories for each gesture and began the collection process.

### Week 3:
- Data Preprocessing: Converted the collected images to grayscale and resized them to a uniform size (64x64).
- preprocess_images.py: Developed a script to preprocess images by converting them to grayscale and resizing.

### Week 4:
- Data Preparation: Loaded the preprocessed images, labeled them, and split the dataset into training and testing sets.
- prepare_data.py: Created a script to organize, label, and split the data into training and testing sets (80/20 split).

### Week 5:
- Model Design: Built a Convolutional Neural Network (CNN) model using TensorFlow and Keras for gesture recognition.
- Model Training: Trained the model using the collected and preprocessed gesture data.
- model.py: Created the script to define, train, and save the trained model.

### Week 6:
- Real-Time Gesture Recognition: Implemented a real-time gesture recognition system using the webcam.
- real_time_recognition.py: Developed a script to capture gestures in real-time and predict hand gestures using the trained model.

### Week 7:
- UI Enhancement: Added on-screen buttons for arithmetic operations (+, -, *, /), allowing users to perform calculations with recognized gestures.
- Operation Implementation: Enhanced the system to allow users to select operations via clickable buttons, and perform calculations using recognized hand gestures.

### Week 8:
- Process Rework: Reworked the process modeling by integrating the preprocessing and loading of images directly into model.py.

### Week 9:
- Data Collection (Part 2): Focused on improving model accuracy by collecting and adding more diverse images of hand gestures.

### Week 10:
- Model Optimization: Continued working on improving accuracy by implementing data augmentation, a learning rate scheduler, and other advanced techniques to enhance the model's training process.