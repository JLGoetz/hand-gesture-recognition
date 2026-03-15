try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    print("MediaPipe version:", mp.__version__)
    print("Tasks API successfully imported!")
    
    # Test if the BaseOptions class is available
    options = python.BaseOptions(model_asset_path='tasks/hand_landmarker.task')
    print("BaseOptions initialized successfully.")
    
except Exception as e:
    print(f"An error occurred: {e}")