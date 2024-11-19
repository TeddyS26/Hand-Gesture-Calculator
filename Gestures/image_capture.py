import cv2
import os

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_available_index(directory):
    existing_files = [f for f in os.listdir(directory) if f.startswith("image_") and f.endswith(".jpg")]
    if not existing_files:
        return 0
    existing_indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    return max(existing_indices) + 1

def capture_images(gesture, num_images):
    # Create a directory for the gesture images
    gesture_dir = f'Gestures/gesture_{gesture}'
    create_directory(gesture_dir)
    
    # Count existing images to start numbering from the next available number
    existing_images = len(os.listdir(gesture_dir))
    count = get_next_available_index(gesture_dir)
    
    cap = cv2.VideoCapture(0)
    
    while count < existing_images + num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Display the frame
        cv2.imshow('Capture Hand Gesture', frame)
        
        # Save the captured image with an incremented count
        img_path = os.path.join(gesture_dir, f'image_{count}.jpg')
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1
        
        # Press 'q' to quit the capture early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    current_gesture = int(input("Enter the gesture number (0-9) to capture: "))
    num_images = int(input("Enter the number of images to capture: "))
    print(f"Capturing images for gesture {current_gesture}")
    capture_images(current_gesture, num_images)