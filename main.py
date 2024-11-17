import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
model = load_model('hand_gesture_model.keras')

IMG_SIZE = 128

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to store the numbers and operation
first_num = None
second_num = None
operation = None
result = None
button_clicked = False

# Function to check if a point is inside a rectangle
def is_inside(x, y, x1, y1, x2, y2):
    return x1 <= x <= x2 and y1 <= y <= y2

# Mouse callback function to detect button clicks
def button_click(event, x, y, flags, param):
    global operation, button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which button is clicked
        if is_inside(x, y, 10, 400, 100, 450):  # "+" button
            operation = '+'
            button_clicked = True
        elif is_inside(x, y, 120, 400, 210, 450):  # "-" button
            operation = '-'
            button_clicked = True
        elif is_inside(x, y, 230, 400, 320, 450):  # "*" button
            operation = '*'
            button_clicked = True
        elif is_inside(x, y, 340, 400, 430, 450):  # "/" button
            operation = '/'
            button_clicked = True

# Create a window and set a mouse callback function
cv2.namedWindow("Gesture Recognition (Colored)")
cv2.setMouseCallback("Gesture Recognition (Colored)", button_click)

while True:
    ret, frame = cap.read()
    
    # Process the frame for prediction (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE))
    reshaped = resized.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    
    # Predict the gesture
    prediction = model.predict(reshaped)
    predicted_class = np.argmax(prediction)

    # Show the regular colored frame
    cv2.putText(frame, f'Gesture: {predicted_class}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw buttons for operations
    if button_clicked:
        # Highlight the clicked button
        cv2.rectangle(frame, (10, 400), (100, 450), (0, 255, 0), -1) if operation == '+' else cv2.rectangle(frame, (10, 400), (100, 450), (255, 0, 0), 2)
        cv2.rectangle(frame, (120, 400), (210, 450), (0, 255, 0), -1) if operation == '-' else cv2.rectangle(frame, (120, 400), (210, 450), (255, 0, 0), 2)
        cv2.rectangle(frame, (230, 400), (320, 450), (0, 255, 0), -1) if operation == '*' else cv2.rectangle(frame, (230, 400), (320, 450), (255, 0, 0), 2)
        cv2.rectangle(frame, (340, 400), (430, 450), (0, 255, 0), -1) if operation == '/' else cv2.rectangle(frame, (340, 400), (430, 450), (255, 0, 0), 2)
    else:
        # Default button display
        cv2.rectangle(frame, (10, 400), (100, 450), (255, 0, 0), 2)  # "+" button
        cv2.putText(frame, "+", (40, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.rectangle(frame, (120, 400), (210, 450), (255, 0, 0), 2)  # "-" button
        cv2.putText(frame, "-", (150, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.rectangle(frame, (230, 400), (320, 450), (255, 0, 0), 2)  # "*" button
        cv2.putText(frame, "*", (260, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.rectangle(frame, (340, 400), (430, 450), (255, 0, 0), 2)  # "/" button
        cv2.putText(frame, "/", (370, 435), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the result
    if result is not None:
        cv2.putText(frame, f'Result: {result}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Gesture Recognition (Colored)', frame)

    # Logic to capture numbers and perform the operation
    if first_num is None:
        first_num = predicted_class
    elif operation is not None and second_num is None:
        second_num = predicted_class
        if operation == '+':
            result = first_num + second_num
        elif operation == '-':
            result = first_num - second_num
        elif operation == '*':
            result = first_num * second_num
        elif operation == '/':
            result = first_num / second_num if second_num != 0 else "Error (div by 0)"
        
        # Reset after showing the result
        first_num = None
        second_num = None
        operation = None
        button_clicked = False
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()