import sys
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# Load the trained model
model = load_model('Models/hand_gesture_model_mediapipe_v2.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

class HandGestureApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Variables to store numbers, operation, and result
        self.first_num = None
        self.second_num = None
        self.operation = None
        self.result = None
        self.gesture_timer = None
        self.stage = "select_operation"

        # Setup UI
        self.setWindowTitle("Hand Gesture Calculator")
        self.setGeometry(100, 100, 800, 600)

        # Camera feed display
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(10, 10, 640, 480)

        # Gesture display
        self.gesture_label = QLabel(self)
        self.gesture_label.setGeometry(10, 500, 300, 30)
        self.gesture_label.setStyleSheet("font-size: 16px; color: green;")

        # Equation display
        self.equation_label = QLabel(self)
        self.equation_label.setGeometry(320, 500, 300, 30)
        self.equation_label.setStyleSheet("font-size: 16px; color: blue;")

        # Result display
        self.result_label = QLabel(self)
        self.result_label.setGeometry(10, 540, 300, 30)
        self.result_label.setStyleSheet("font-size: 16px; color: red;")

        # Buttons for operations
        self.add_button = QPushButton("+", self)
        self.add_button.setGeometry(660, 10, 100, 50)
        self.add_button.clicked.connect(lambda: self.select_operation("+"))

        self.subtract_button = QPushButton("-", self)
        self.subtract_button.setGeometry(660, 70, 100, 50)
        self.subtract_button.clicked.connect(lambda: self.select_operation("-"))

        self.multiply_button = QPushButton("*", self)
        self.multiply_button.setGeometry(660, 130, 100, 50)
        self.multiply_button.clicked.connect(lambda: self.select_operation("*"))

        self.divide_button = QPushButton("/", self)
        self.divide_button.setGeometry(660, 190, 100, 50)
        self.divide_button.clicked.connect(lambda: self.select_operation("/"))

        self.clear_button = QPushButton("AC", self)
        self.clear_button.setGeometry(660, 250, 100, 50)
        self.clear_button.clicked.connect(self.clear_all)

        # Start the camera feed
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def select_operation(self, operation):
        if self.stage == "select_operation":
            self.operation = operation
            self.stage = "first_number"
            self.gesture_timer = cv2.getTickCount()

    def clear_all(self):
        self.first_num = None
        self.second_num = None
        self.operation = None
        self.result = None
        self.stage = "select_operation"
        self.gesture_timer = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        result_mediapipe = hands.process(rgb_frame)

        # Predict gesture
        current_gesture = None
        if result_mediapipe.multi_hand_landmarks:
            lm_combined = []
            for hand_landmarks in result_mediapipe.multi_hand_landmarks:
                lm = []
                for point in hand_landmarks.landmark:
                    lm.extend([point.x, point.y, point.z])
                lm_combined.extend(lm)
            # If fewer than 126 features, pad with zeros
            while len(lm_combined) < 126:
                lm_combined.extend([0.0] * 3)

            # Convert landmarks to numpy array and predict gesture
            landmarks = np.array(lm_combined).reshape(1, -1)
            prediction = model.predict(landmarks)
            current_gesture = np.argmax(prediction)

        # Manage stages
        if self.stage == "first_number" and self.gesture_timer:
            elapsed_time = (cv2.getTickCount() - self.gesture_timer) / cv2.getTickFrequency()
            if elapsed_time >= 5:
                self.first_num = current_gesture
                self.stage = "second_number"
                self.gesture_timer = cv2.getTickCount()

        elif self.stage == "second_number" and self.gesture_timer:
            elapsed_time = (cv2.getTickCount() - self.gesture_timer) / cv2.getTickFrequency()
            if elapsed_time >= 5:
                self.second_num = current_gesture
                if self.operation and self.first_num is not None and self.second_num is not None:
                    # Perform the calculation
                    if self.operation == '+':
                        self.result = self.first_num + self.second_num
                    elif self.operation == '-':
                        self.result = self.first_num - self.second_num
                    elif self.operation == '*':
                        self.result = self.first_num * self.second_num
                    elif self.operation == '/':
                        self.result = self.first_num / self.second_num if self.second_num != 0 else "Error (div by 0)"
                self.stage = "show_result"
                self.gesture_timer = None

        # Update the labels
        self.gesture_label.setText(f"Gesture: {current_gesture}" if current_gesture is not None else "Gesture: None")
        self.equation_label.setText(
            f"{self.first_num if self.first_num is not None else '___'} {self.operation if self.operation else ''} {self.second_num if self.second_num is not None else '___'}"
        )
        self.result_label.setText(f"Result: {self.result}" if self.result is not None else "Result: ___")

        # Convert the frame to QImage and display it
        qt_frame = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_frame)
        self.camera_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandGestureApp()
    window.show()
    sys.exit(app.exec_())
