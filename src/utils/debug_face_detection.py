import cv2
import numpy as np
import os

def debug_file_search():
    print("Current working directory:", os.getcwd())
    
    # Check if data directory exists
    data_path = "../../data/raw/faceforensics"
    print(f"Looking for data at: {data_path}")
    print(f"Data path exists: {os.path.exists(data_path)}")
    
    if os.path.exists(data_path):
        print("Contents of faceforensics directory:")
        for root, dirs, files in os.walk(data_path):
            print(f"Directory: {root}")
            print(f"  Subdirs: {dirs}")
            print(f"  Files: {[f for f in files if f.endswith('.mp4')][:3]}")  # Show first 3 mp4 files
            if files:
                break
    
    # Try to find a specific video
    test_video = "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/183.mp4"
    print(f"\nChecking specific video: {test_video}")
    print(f"Video exists: {os.path.exists(test_video)}")
    
    if os.path.exists(test_video):
        # Test if OpenCV can read it
        cap = cv2.VideoCapture(test_video)
        ret, frame = cap.read()
        print(f"Can read video: {ret}")
        if ret:
            print(f"Frame shape: {frame.shape}")
        cap.release()

def test_face_detection_on_specific_video():
    video_path = "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/183.mp4"
    
    if not os.path.exists(video_path):
        print("Video not found, skipping face detection test")
        return
    
    print(f"\nTesting face detection on: {video_path}")
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print(f"Face cascade loaded: {not face_cascade.empty()}")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_faces = 0
    
    for i in range(5):  # Check first 5 frames
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {i}")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        print(f"Frame {i}: {len(faces)} faces detected")
        total_faces += len(faces)
        frame_count += 1
    
    cap.release()
    print(f"Total faces found: {total_faces} across {frame_count} frames")

if __name__ == "__main__":
    debug_file_search()
    test_face_detection_on_specific_video()
