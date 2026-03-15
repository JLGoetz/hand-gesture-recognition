import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.neighbors import KNeighborsClassifier

# --- 1. ML CLASSIFIER ---
class MLGestureClassifier:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        
    def extract_robust_features(self, hand_landmarks):
        # Must match the logic in collector.py exactly!
        wrist = hand_landmarks[0]
        middle_tip = hand_landmarks[12]
        
        scale = np.sqrt((middle_tip.x - wrist.x)**2 + 
                        (middle_tip.y - wrist.y)**2 + 
                        (middle_tip.z - wrist.z)**2)
        
        if scale == 0: scale = 1.0
        
        features = []
        for lm in hand_landmarks:
            features.extend([(lm.x - wrist.x) / scale, 
                             (lm.y - wrist.y) / scale, 
                             (lm.z - wrist.z) / scale])
        return features

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def classify(self, hand_landmarks):
        if not self.is_trained: return "UNTRAINED"
        features = np.array(self.extract_robust_features(hand_landmarks)).reshape(1, -1)
        return self.model.predict(features)[0]

# --- 2. ACTION MANAGER ---
class ActionManager:
    def __init__(self, debounce=0.7):
        self.debounce = debounce
        self.last_action_time = {}
        self.mapping = {
            'Left': {"FIST": "LEFT_GRAB", "PEACE": "LEFT_ROTATE"},
            'Right': {"FIST": "RIGHT_GRAB", "PEACE": "RIGHT_ROTATE"}
        }

    def process(self, hand_label, gesture):
        now = time.time()
        action = self.mapping.get(hand_label, {}).get(gesture)
        if action:
            if now - self.last_action_time.get(hand_label, 0) > self.debounce:
                self.last_action_time[hand_label] = now
                return action
        return None

# --- 3. CONFIG & INIT ---
MODEL_PATH = 'tasks/hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

ml_classifier = MLGestureClassifier()
action_manager = ActionManager(debounce=0.7)

cap = cv2.VideoCapture(0)

# --- 4. MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            label = detection_result.handedness[i][0].category_name
            
            # Classify and Manage
            gesture = ml_classifier.classify(hand_landmarks)
            action = action_manager.process(label, gesture)
            
            if action:
                print(f"Triggered: {action}")
            
            # UI Feedback
            cv2.putText(frame, f"{label}: {gesture}", (50, 50 + (i*50)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()