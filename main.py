import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# 1. Configuration
MODEL_PATH = os.path.join('tasks', 'hand_landmarker.task')

# 2. Initialize Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

#buffer for smoothing
class GestureBuffer:
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)
        
    def add_gesture(self, status):
        self.buffer.append(status)
        
    def get_smoothed_status(self):
        if len(self.buffer) < self.buffer.maxlen:
            return None # Not enough data yet
        
        # Determine the most frequent gesture in the buffer
        # status is a list of 5 booleans [thumb, index, middle, ring, pinky]
        # We average each finger across the last N frames
        avg_status = []
        for i in range(5):
            count_true = sum(1 for frame in self.buffer if frame[i])
            avg_status.append(count_true / len(self.buffer) > 0.5)
        return avg_status
    
# 3. Finger Logic Function
def get_finger_status(hand_landmarks):
    """
    Returns a list of 5 booleans [thumb, index, middle, ring, pinky]
    True = finger is extended (up), False = finger is folded (down)
    """
    # Landmark indices for Tips [4, 8, 12, 16, 20] and PIP/IP joints [3, 6, 10, 14, 18]
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    
    fingers = []
    
    # Thumb (Checking x-coordinate relative to IP joint)
    # Note: Logic depends on palm orientation
    fingers.append(hand_landmarks[tips[0]].x > hand_landmarks[pips[0]].x)
    
    # Other 4 fingers (Comparing y-coordinate)
    for i in range(1, 5):
        fingers.append(hand_landmarks[tips[i]].y < hand_landmarks[pips[i]].y)
        
    return fingers

# 

# 4. Main Camera Loop
cap = cv2.VideoCapture(0)

# Initialize buffer outside the loop
gesture_buffer = GestureBuffer(size=10)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect
    detection_result = detector.detect(mp_image)

    # Draw and Analyze
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Get raw status
            raw_status = get_finger_status(hand_landmarks)
            gesture_buffer.add_gesture(raw_status)
            
            # Get smoothed status
            status = gesture_buffer.get_smoothed_status()
            
            if status: # Only proceed if we have enough frames
                if all(status):
                    cv2.putText(frame, "HAND OPEN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif not any(status):
                    cv2.putText(frame, "FIST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
