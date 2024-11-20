import cv2
import os
import mediapipe as mp
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

def extract_landmarks_from_images():
    landmark_data = {}
    for gesture in range(10):
        gesture_dir = f'./Gestures/gesture_{gesture}'
        if not os.path.exists(gesture_dir):
            print(f"Directory {gesture_dir} does not exist. Skipping.")
            continue
        
        print(f"Processing images for gesture {gesture}...")
        landmarks = []
        for img_name in os.listdir(gesture_dir):
            img_path = os.path.join(gesture_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}. Skipping.")
                continue
            
            # Convert to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Hands
            result = hands.process(rgb_img)
            
            # Initialize combined landmarks for two hands
            lm_combined = []
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    lm = []
                    for point in hand_landmarks.landmark:
                        lm.extend([point.x, point.y, point.z])
                    lm_combined.extend(lm)
            
            # Ensure exactly 126 features (21 points × 3 dimensions × 2 hands)
            while len(lm_combined) < 126:
                lm_combined.extend([0.0] * 3)  # Add zero padding for missing points
            
            # Only append if the shape is correct
            if len(lm_combined) == 126:
                landmarks.append(lm_combined)
        
        # Save landmarks for this gesture
        landmark_data[gesture] = landmarks
        print(f"Extracted {len(landmarks)} landmarks for gesture {gesture}")
    
    # Save all landmarks to a pickle file
    with open('./Landmarks/gesture_landmarks_v2.pkl', 'wb') as f:
        pickle.dump(landmark_data, f)
    print("All landmarks saved to gesture_landmarks_v2.pkl")

if __name__ == "__main__":
    extract_landmarks_from_images()
