import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('hand_gesture_model.keras')

IMG_SIZE = 64

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture the frame
    
    # Process the frame for prediction (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    reshaped = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    
    # Predict the gesture
    prediction = model.predict(reshaped)
    predicted_class = np.argmax(prediction)
    
    # Display the predicted gesture on the colored frame
    cv2.putText(frame, f'Gesture: {predicted_class}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the regular colored frame (with the prediction result)
    cv2.imshow('Gesture Recognition (Colored)', frame)
    
    # Show the grayscale frame (to visualize what the model sees)
    cv2.imshow('Processed Image (Grayscale)', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()