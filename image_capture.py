import cv2
import os

# Set up directories for each gesture (0-9)
gesture_folders = [f'gesture_{i}' for i in range(11)]
for folder in gesture_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize the webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit and 's' to save images")

# Specify which digit to capture (modify this for each run)
current_gesture = 6  # Change this to capture images for different gestures

image_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # Save the frame when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        image_path = f'gesture_{current_gesture}/image_{image_count}.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")
        image_count += 1

    # Quit the capture when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
