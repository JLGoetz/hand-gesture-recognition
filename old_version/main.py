import cv2
import os
import math 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# 1. Configuration
MODEL_PATH = os.path.join('tasks', 'hand_landmarker.task')

# 2. Initialize Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2) #alows for 2 hands to be rcognizes
detector = vision.HandLandmarker.create_from_options(options)

#buffer for smoothing
class GestureManager:
    def __init__(self, size=7):
        self.size = size
        # Stores buffers as: {'Left': deque(...), 'Right': deque(...)}
        self.buffers = {}

    def add_gesture(self, hand_label, status):
        # Check if this hand has a buffer yet
        if hand_label not in self.buffers:
            self.buffers[hand_label] = deque(maxlen=self.size)
        # Add the status to this specific hand's buffer
        self.buffers[hand_label].append(status)
        
    def get_smoothed_status(self, hand_label):
        # Retrieve the specific buffer for this hand
        buffer = self.buffers.get(hand_label)
        if not buffer:
            return None
        
        # Calculate smoothed state
        avg_status = []
        for i in range(5):
            # Use 'buffer' instead of 'self.buffer'
            count_true = sum(1 for frame in buffer if frame[i])
            # Use 'len(buffer)' instead of 'len(self.buffer)'
            avg_status.append(count_true / len(buffer) > 0.5)
        return avg_status
    
# 3. Finger Logic Function
def get_finger_status(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    wrist = hand_landmarks[0]
    
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    fingers = []
    
    # --- BETTER THUMB LOGIC ---
    # Compare Thumb Tip to Pinky Base (17). 
    # If the thumb is tucked in (Fist), it's close to the pinky base.
    thumb_to_pinky_base = dist(hand_landmarks[4], hand_landmarks[17])
    thumb_joint_to_pinky_base = dist(hand_landmarks[2], hand_landmarks[17])
    
    # If Tip is further from pinky than the joint, it's extended
    fingers.append(thumb_to_pinky_base > thumb_joint_to_pinky_base)

    # --- OTHER 4 FINGERS ---
    for i in range(1, 5):
        tip_dist = dist(hand_landmarks[tips[i]], wrist)
        pip_dist = dist(hand_landmarks[pips[i]], wrist)
        fingers.append(tip_dist > pip_dist)
        
    return fingers

# 3. a classify_gesture helper funciton for cleaner handleing of mulit-hand combinatins
def classify_gesture(status):
    if not status: return ""
    
    # [thumb, index, middle, ring, pinky]
    if all(status): return "OPEN"
    if not any(status): return "FIST"

    # Peace: Index and Middle up, others down
    if not status[0] and status[1] and status[2] and not status[3] and not status[4]:
        return "PEACE"

    # Thumbs Up: Only thumb up
    if status[0] and not any(status[1:]):
        return "THUMBS UP"
        
    # Rock On: Index and Pinky up
    if status[1] and status[4] and not status[2] and not status[3]:
        return "ROCK ON"
        
    return "UNKNOWN"


# 4. Main Camera Loop
cap = cv2.VideoCapture(0)

# Initialize buffer outside the loop
gesture_manager = GestureManager(size=10)

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
       h, w, _ = frame.shape
    
    # Store statuses to evaluate "combination" gestures later
    current_frame_gestures = {} 

    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        # Determine handedness: Note that MediaPipe labels hands from the perspective 
        # of the hand itself, not the camera.
        hand_label = detection_result.handedness[i][0].category_name 
        
        # Draw landmarks
        for lm in hand_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (255, 0, 0), -1)

        # 2. Update the specific buffer for this hand
        raw_status = get_finger_status(hand_landmarks)
        gesture_manager.add_gesture(hand_label, raw_status)
        
        # 3. Get smoothed status for this specific hand
        status = gesture_manager.get_smoothed_status(hand_label)
        current_frame_gestures[hand_label] = status
        # Right after current_frame_gestures[hand_label] = status to view what sensors see
        #print(f"DEBUG: {hand_label} current status: {status}")
        
        # 4. Display status next to the specific hand
        if status:
            # Simple check for individual hands
            text = "OPEN" if all(status) else "FIST" if not any(status) else ""
            cv2.putText(frame, f"{hand_label}: {text}", (int(hand_landmarks[0].x * w), 
                        int(hand_landmarks[0].y * h) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. Combination Logic (Outside the loop)
    l_status = current_frame_gestures.get('Left')
    r_status = current_frame_gestures.get('Right')

    if l_status and r_status:
        # Get readable names for the gestures
        l_gesture = classify_gesture(l_status)
        r_gesture = classify_gesture(r_status)
        
        # Display individual statuses for debugging
        cv2.putText(frame, f"L: {l_gesture} | R: {r_gesture}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Combination 1: Double Peace
        if l_gesture == "PEACE" and r_gesture == "PEACE":
            cv2.putText(frame, "DOUBLE PEACE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Combination 2: Double Fist (Brace for impact?)
        elif l_gesture == "FIST" and r_gesture == "FIST":
            cv2.putText(frame, "BOXING STANCE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Combination 3: Prayer / Namaste (Both Open)
        elif l_gesture == "OPEN" and r_gesture == "OPEN":
            cv2.putText(frame, "PRAYER / GREETING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    
    cv2.imshow('Hand Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
