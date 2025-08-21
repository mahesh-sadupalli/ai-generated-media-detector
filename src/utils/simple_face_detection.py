import cv2
import numpy as np
from typing import List, Tuple
import os

class SimpleFaceDetector:
    def __init__(self):
        """Initialize using Haar Cascade face detector"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def extract_faces_from_video(self, video_path: str, max_frames: int = 5) -> List[np.ndarray]:
        """Extract faces from video frames"""
        cap = cv2.VideoCapture(video_path)
        faces = []
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            detected_faces = self.detect_faces(frame)
            for (x, y, w, h) in detected_faces:
                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    face_resized = cv2.resize(face, (224, 224))
                    faces.append(face_resized)
                    
            frame_count += 1
        
        cap.release()
        return faces

def test_simple_face_detection():
    detector = SimpleFaceDetector()
    
    # Test on specific videos we know exist
    test_videos = [
        "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/183.mp4",
        "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/585.mp4"
    ]
    
    total_faces = 0
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            print(f"Testing: {os.path.basename(video_path)}")
            faces = detector.extract_faces_from_video(video_path, max_frames=3)
            print(f"  Extracted {len(faces)} faces")
            total_faces += len(faces)
        else:
            print(f"Video not found: {video_path}")
    
    print(f"\nTotal faces extracted: {total_faces}")
    
    if total_faces > 0:
        print("✓ Face detection working successfully!")
        return True
    else:
        print("✗ No faces detected")
        return False

if __name__ == "__main__":
    test_simple_face_detection()
