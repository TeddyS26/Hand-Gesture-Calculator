import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 64

data = []
labels = []

# Load images and their corresponding labels
for i in range(10):
    folder = f'gesture_{i}'
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        print("Loading", img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(i)

data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
data = data / 255.0
labels = np.array(labels)

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

print("Data loaded and split into training and testing sets.")