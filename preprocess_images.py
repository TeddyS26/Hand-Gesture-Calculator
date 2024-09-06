import cv2
import os

def preprocess_images(folder):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize the image to a fixed size
        resized_img = cv2.resize(gray_img, (64, 64))
        
        # Save the preprocessed image
        cv2.imwrite(img_path, resized_img)

# Preprocess images for all gestures
for i in range(10):
    preprocess_images(f'gesture_{i}')
