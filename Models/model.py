import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Constants
EPOCHS = 50
BATCH_SIZE = 32
NUM_LANDMARKS = 126

# Load landmark data
def load_landmark_data():
    with open('./Landmarks/gesture_landmarks_v3.pkl', 'rb') as f:
        landmark_data = pickle.load(f)
    
    data = []
    labels = []
    for gesture, landmarks in landmark_data.items():
        data.extend(landmarks)
        labels.extend([gesture] * len(landmarks))
    
    data = np.array(data)
    labels = np.array(labels)
    return train_test_split(data, labels, test_size=0.1, random_state=42)

X_train, X_test, y_train, y_test = load_landmark_data()
print("Landmark data loaded and split into training and testing sets.")

# Build the model
model = Sequential([
    Input(shape=(NUM_LANDMARKS,)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Model built and compiled.")

# Learning rate scheduler
def scheduler(epoch, lr):
    min_lr = 1e-5
    if epoch < 10:
        return lr
    else:
        new_lr = lr * tf.math.exp(-0.1)
        return float(max(new_lr, min_lr))

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model
model.save('./Models/hand_gesture_model_mediapipe_v3.keras')
print("Model trained and saved.")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('./Plots/training_validation_accuracy.png')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('./Plots/training_validation_loss.png')
plt.show()

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list(range(10)))
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.savefig('./Plots/confusion_matrix.png')
plt.show()