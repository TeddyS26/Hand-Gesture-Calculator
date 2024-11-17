import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

IMG_SIZE = 128
EPOCHS = 50
BATCH_SIZE = 32

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.6, 1.4],
    shear_range=0.2,
    zoom_range=0.3,
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

# Convert grayscale images to three channels for MobileNetV2
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

# Using a pre-trained MobileNetV2 model for transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Build the full model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model with an appropriate learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Transfer learning model built and compiled.")

# Learning rate scheduler with minimum threshold
def scheduler(epoch, lr):
    min_lr = 1e-5
    if epoch < 10:
        return lr
    else:
        new_lr = lr * tf.math.exp(-0.1)
        return float(max(new_lr, min_lr))

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the Model
history = model.fit(
    datagen.flow(X_train_rgb, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test_rgb, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Save the Model
model.save('hand_gesture_model.keras')
print("Transfer learning model trained and saved.")

# Evaluate the Model
test_loss, test_acc = model.evaluate(X_test_rgb, y_test)
print(f'Test Accuracy: {test_acc}')
