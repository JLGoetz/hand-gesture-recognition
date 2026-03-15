import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.neighbors import KNeighborsClassifier
from collections import deque


# --- 1. ML CLASSIFIER ---
class MLGestureClassifier:
    def __init__(self, threshold=0.7, buffer_size=5):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        self.threshold = threshold # Confidence limit
        self.buffer_size = buffer_size
        # Store a buffer of recent results for each hand
        self.prediction_buffers = {'Left': deque(maxlen=buffer_size), 
                                   'Right': deque(maxlen=buffer_size)}
        
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

    def classify(self, hand_landmarks, hand_label):
        if not self.is_trained: return "UNTRAINED"
        
        features = np.array(self.extract_robust_features(hand_landmarks)).reshape(1, -1)
        
        # Get probabilities for all classes
        probs = self.model.predict_proba(features)[0]
        max_prob = np.max(probs)
        gesture = self.model.classes_[np.argmax(probs)]

        # Add to buffer
        self.prediction_buffers[hand_label].append((gesture, max_prob))
        
        # Calculate average confidence
        avg_probs = {}
        for g, p in self.prediction_buffers[hand_label]:
            avg_probs[g] = avg_probs.get(g, 0) + (p / self.buffer_size)
            
        # Get the winner of the smoothed buffer
        best_gesture = max(avg_probs, key=avg_probs.get)
        best_confidence = avg_probs[best_gesture]
        
        return best_gesture if best_confidence >= self.threshold else "UNKNOWN"

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
    
    # 1. Always initialize overlay at the start of the frame loop
    overlay = frame.copy() 
    
    # Process image for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        y_offset = 50
        
        # 2. Iterate through all detected hands
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            label = detection_result.handedness[i][0].category_name
            
            # Classify
            gesture = ml_classifier.classify(hand_landmarks, label)
            
            # Determine UI Color
            color = (128, 128, 128) if "UNKNOWN" in gesture else (0, 255, 0)
            
            # Process Action
            if "UNKNOWN" not in gesture:
                action = action_manager.process(label, gesture)
                if action:
                    print(f"Triggered: {action}")
            
            # 3. Draw UI (Rectangle on overlay, Text on frame)
            cv2.rectangle(overlay, (40, y_offset - 35), (380, y_offset + 15), (0, 0, 0), -1)
            cv2.putText(frame, f"{label}: {gesture}", (50, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            y_offset += 50
    
    # 4. Blend the overlay with the frame (outside the if-block)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.imshow('Hand Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()