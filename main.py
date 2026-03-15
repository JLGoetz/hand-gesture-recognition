import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import numpy as np
# You'll need: pip install scikit-learn
from sklearn.neighbors import KNeighborsClassifier 

# --- NEW: ML HELPER ---
def extract_features(hand_landmarks):
    # Flatten 21 landmarks (x,y,z) into 63 features relative to wrist
    wrist = hand_landmarks[0]
    return [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in hand_landmarks]

class MLGestureClassifier:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        # For this example, we assume you have a 'gestures.csv' 
        # or a pre-trained model. For now, let's use a placeholder.
        self.is_trained = False

    def classify(self, landmarks):
        if not self.is_trained: return "UNTRAINED"
        features = np.array(extract_features(landmarks)).flatten().reshape(1, -1)
        return self.model.predict(features)[0]

# --- YOUR EXISTING CONFIG ---
MODEL_PATH = os.path.join('tasks', 'hand_landmarker.task')
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize ML Classifer
ml_manager = MLGestureClassifier()

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        h, w, _ = frame.shape
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            hand_label = detection_result.handedness[i][0].category_name 
            
            # Draw landmarks
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1)

            # ML CLASSIFICATION STEP
            gesture_name = ml_manager.classify(hand_landmarks)
            
            # UI
            cv2.putText(frame, f"{hand_label}: {gesture_name}", 
                        (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('ML Hand Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()