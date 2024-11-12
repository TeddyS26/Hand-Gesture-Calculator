import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

IMG_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 32

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to load and preprocess data
def load_and_preprocess_data():
    data = []
    labels = []
    for i in range(10):
        folder = f'gesture_{i}'
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(i)
    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    labels = np.array(labels)
    return train_test_split(data, labels, test_size=0.1, random_state=42)

X_train, X_test, y_train, y_test = load_and_preprocess_data()
print("Data loaded and split into training and testing sets.")

# Model Definition
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Model built and compiled.")

# Learning rate scheduler with minimum threshold
def scheduler(epoch, lr):
    min_lr = 1e-5  # Set a minimum learning rate threshold
    if epoch < 10:
        return lr
    else:
        new_lr = lr * tf.math.exp(-0.1)
        return float(max(new_lr, min_lr))  # Ensure the learning rate doesn't fall below min_lr

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the Model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Save the Model
model.save('hand_gesture_model.keras')
print("Model trained and saved.")

# Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
