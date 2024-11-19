import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import time

# Load the trained model
model = load_model('Models/hand_gesture_model_mediapipe.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

IMG_SIZE = 128

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to store the numbers, operation, and result
first_num = None
second_num = None
operation = None
result = None
button_clicked = False

# Timers
gesture_timer = None
stage = "select_operation"

# Function to check if a point is inside a rectangle
def is_inside(x, y, x1, y1, x2, y2):
    return x1 <= x <= x2 and y1 <= y <= y2

# Mouse callback function to detect button clicks
def button_click(event, x, y, flags, param):
    global operation, button_clicked, first_num, second_num, result, stage, gesture_timer
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which button is clicked
        if is_inside(x, y, 10, 400, 100, 450):  # "+" button
            operation = '+'
            button_clicked = True
            stage = "first_number"
            gesture_timer = time.time()
        elif is_inside(x, y, 120, 400, 210, 450):  # "-" button
            operation = '-'
            button_clicked = True
            stage = "first_number"
            gesture_timer = time.time()
        elif is_inside(x, y, 230, 400, 320, 450):  # "*" button
            operation = '*'
            button_clicked = True
            stage = "first_number"
            gesture_timer = time.time()
        elif is_inside(x, y, 340, 400, 430, 450):  # "/" button
            operation = '/'
            button_clicked = True
            stage = "first_number"
            gesture_timer = time.time()
        elif is_inside(x, y, 460, 400, 550, 450):  # "AC" button
            first_num = None
            second_num = None
            operation = None
            result = None
            stage = "select_operation"
            button_clicked = False
            gesture_timer = None

# Create a window and set a mouse callback function
cv2.namedWindow("Hand Gesture Calculator")
cv2.setMouseCallback("Hand Gesture Calculator", button_click)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Hands
    result_mediapipe = hands.process(rgb_frame)

    # Predict gesture
    current_gesture = None
    if result_mediapipe.multi_hand_landmarks:
        for hand_landmarks in result_mediapipe.multi_hand_landmarks:
            # Extract 21 landmarks
            landmarks = []
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

            # Convert landmarks to numpy array
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(landmarks)
            current_gesture = np.argmax(prediction)

    # Manage stages
    if stage == "first_number" and gesture_timer:
        elapsed_time = time.time() - gesture_timer
        if elapsed_time >= 5:
            first_num = current_gesture
            stage = "second_number"
            gesture_timer = time.time()  # Reset the timer
    elif stage == "second_number" and gesture_timer:
        elapsed_time = time.time() - gesture_timer
        if elapsed_time >= 5:
            second_num = current_gesture
            if operation and first_num is not None and second_num is not None:
                # Perform the calculation
                if operation == '+':
                    result = first_num + second_num
                elif operation == '-':
                    result = first_num - second_num
                elif operation == '*':
                    result = first_num * second_num
                elif operation == '/':
                    result = first_num / second_num if second_num != 0 else "Error (div by 0)"
            stage = "show_result"
            gesture_timer = None

    # Display the current gesture
    if current_gesture is not None:
        cv2.putText(frame, f'Gesture: {current_gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display operation and numbers
    if operation is None:
        operation_text = "___ __ ___"
    else:
        operation_text = f"{first_num if first_num is not None else '___'} {operation} {second_num if second_num is not None else '___'}"
    cv2.putText(frame, operation_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display result
    if result is not None and stage == "show_result":
        cv2.putText(frame, f'Result: {result}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw buttons
    cv2.rectangle(frame, (10, 400), (100, 450), (255, 0, 0), 2)  # "+" button
    cv2.putText(frame, "+", (40, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(frame, (120, 400), (210, 450), (255, 0, 0), 2)  # "-" button
    cv2.putText(frame, "-", (150, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(frame, (230, 400), (320, 450), (255, 0, 0), 2)  # "*" button
    cv2.putText(frame, "*", (260, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(frame, (340, 400), (430, 450), (255, 0, 0), 2)  # "/" button
    cv2.putText(frame, "/", (370, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(frame, (460, 400), (550, 450), (255, 0, 0), 2)  # "AC" button
    cv2.putText(frame, "AC", (480, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Hand Gesture Calculator', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()