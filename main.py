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
    def __init__(self, size=10):
        self.buffer = deque(maxlen=size)
        
    def add_gesture(self, status):
        self.buffer.append(status)
        
    def get_smoothed_status(self):
        # If empty, return None
        if not self.buffer:
            return None
        
        # Calculate current state based on data available (even if not full)
        avg_status = []
        for i in range(5):
            count_true = sum(1 for frame in self.buffer if frame[i])
            # If > 50% of the frames in the buffer say "True", consider it True
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
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 0), -1)

            # 2. SEPARATE logic (smoothing)
            raw_status = get_finger_status(hand_landmarks)
            gesture_buffer.add_gesture(raw_status)
            status = gesture_buffer.get_smoothed_status()

            # 3. Draw text based on smoothed status
            if status is not None:
                # Pattern Logic
                # status = [thumb, index, middle, ring, pinky]
                
                if all(status):
                    text = "HAND OPEN"
                elif not any(status):
                    text = "FIST"
                elif status[1] and status[2] and not status[0] and not status[3] and not status[4]:
                    text = "PEACE SIGN"
                elif status[0] and not status[1] and not status[2] and not status[3] and not status[4]:
                    text = "THUMBS UP"
                elif status[1] and status[4] and not status[2] and not status[3]:
                    text = "ROCK ON"
                else:
                    text = "" # No recognized shape

                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Hand Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
