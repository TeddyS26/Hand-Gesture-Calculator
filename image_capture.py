import cv2
import os

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_images(gesture, num_images=600):
    # Create a directory for the gesture images (same as the original)
    gesture_dir = f'gesture_{gesture}'
    create_directory(gesture_dir)
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Display the frame
        cv2.imshow('Capture Hand Gesture', frame)
        
        # Save the captured image (same save path structure as original)
        img_path = os.path.join(gesture_dir, f'image_{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1
        
        # Press 'q' to quit the capture early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Capture images for a specific gesture
    current_gesture = 9  # Modify this for the gesture you're capturing
    print(f"Capturing images for gesture {current_gesture}")
    capture_images(gesture=current_gesture, num_images=600)