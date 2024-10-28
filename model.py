import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os

IMG_SIZE = 64

# Step 1: Load and Preprocess the Data
data = []
labels = []

# Load images and their corresponding labels
for i in range(10):
    folder = f'gesture_{i}'
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(i)

# Convert data and labels to numpy arrays
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
data = data / 255.0
labels = np.array(labels)

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Step 2: Build the CNN Model
model = Sequential()

# Input layer
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

# Convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output for fully connected layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Model built and compiled.")

# Step 3: Train the Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 4: Save the Model
model.save('hand_gesture_model.keras')

print("Model trained and saved.")

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
