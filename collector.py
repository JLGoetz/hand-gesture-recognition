import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# 1. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=os.path.join('tasks', 'hand_landmarker.task'))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Scale-Invariant Feature Extraction
def extract_robust_features(hand_landmarks):
    """
    Normalizes coordinates by subtracting the wrist position 
    and dividing by the distance to the middle finger tip (Scale Invariance).
    """
    wrist = hand_landmarks[0]
    middle_tip = hand_landmarks[12]
    
    # Calculate scale factor (3D Euclidean distance from wrist to middle tip)
    scale = np.sqrt((middle_tip.x - wrist.x)**2 + 
                    (middle_tip.y - wrist.y)**2 + 
                    (middle_tip.z - wrist.z)**2)
    
    # Safety check to avoid division by zero
    if scale == 0: scale = 1.0
    
    features = []
    for lm in hand_landmarks:
        # Normalize relative to wrist and scale
        features.extend([(lm.x - wrist.x) / scale, 
                         (lm.y - wrist.y) / scale, 
                         (lm.z - wrist.z) / scale])
    return features

# 3. Load or Create Dataset
# Define the path
DATA_DIR = 'training'
DATA_FILE = os.path.join(DATA_DIR, 'training_data.npy')

# Ensure directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Update loading logic
if os.path.exists(DATA_FILE):
    data = np.load(DATA_FILE, allow_pickle=True).item()
else:
    data = {}

label = input("Enter gesture name (e.g., FIST, PEACE): ").upper()
if label not in data: 
    data[label] = []

# 4. Capture Loop with Visual Countdown
cap = cv2.VideoCapture(0)

# Display a 5-second countdown on screen BEFORE capturing
for i in range(5, 0, -1):
    start_time = time.time()
    while time.time() - start_time < 1:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Display the countdown text
        cv2.putText(frame, f"Starting in: {i}", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('Data Collector', frame)
        cv2.waitKey(1)

print(f"Capturing 10 samples for '{label}'...")

count = 0
while count < 10:
    success, frame = cap.read()
    if not success: continue
    
    frame = cv2.flip(frame, 1)
    # Visual feedback: Current progress
    cv2.putText(frame, f"Capturing: {count+1}/10", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Data Collector', frame)
    
    # Process Frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    
    if result.hand_landmarks:
        features = extract_robust_features(result.hand_landmarks[0])
        data[label].append(features)
        count += 1
        cv2.waitKey(1000) # 1 second delay while holding the pose
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()

# 5. Save Data
np.save(DATA_FILE, data)
print(f"\nSuccess! Total samples for '{label}': {len(data[label])}")
cap.release()