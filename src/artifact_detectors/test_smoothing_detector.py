import cv2
import numpy as np
import os
import sys
sys.path.append('../utils')
from simple_face_detection import SimpleFaceDetector
from smoothing_detector import SmoothingArtifactDetector

def test_detector_on_dataset():
    """Test smoothing detector on real vs generated samples"""
    detector = SmoothingArtifactDetector()
    face_extractor = SimpleFaceDetector()
    
    results = {}
    
    # Test on real faces
    print("Testing on REAL faces:")
    real_video = "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/183.mp4"
    if os.path.exists(real_video):
        real_faces = face_extractor.extract_faces_from_video(real_video, max_frames=3)
        real_scores = []
        
        for i, face in enumerate(real_faces):
            score, details = detector.detect_smoothing_artifacts(face)
            real_scores.append(score)
            print(f"  Real face {i}: {score:.3f} - {details['interpretation']}")
        
        results['real'] = np.mean(real_scores)
        print(f"  Average real face score: {results['real']:.3f}")
    
    # Test on generated samples with high pixel loss (should have high smoothing scores)
    print("\nTesting on HIGH_PIXEL artifacts (should detect smoothing):")
    pixel_dir = "../../data/generated/high_pixel"
    if os.path.exists(pixel_dir):
        pixel_scores = []
        pixel_files = [f for f in os.listdir(pixel_dir) if f.endswith('.jpg')][:5]
        
        for filename in pixel_files:
            img_path = os.path.join(pixel_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            score, details = detector.detect_smoothing_artifacts(img)
            pixel_scores.append(score)
            print(f"  {filename}: {score:.3f} - {details['interpretation']}")
        
        results['high_pixel'] = np.mean(pixel_scores)
        print(f"  Average high_pixel score: {results['high_pixel']:.3f}")
    
    # Test on other artifact types (should have lower smoothing scores)
    for artifact_type in ['high_perceptual', 'high_adversarial']:
        print(f"\nTesting on {artifact_type.upper()} artifacts:")
        artifact_dir = f"../../data/generated/{artifact_type}"
        if os.path.exists(artifact_dir):
            artifact_scores = []
            artifact_files = [f for f in os.listdir(artifact_dir) if f.endswith('.jpg')][:3]
            
            for filename in artifact_files:
                img_path = os.path.join(artifact_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                score, details = detector.detect_smoothing_artifacts(img)
                artifact_scores.append(score)
                print(f"  {filename}: {score:.3f}")
            
            results[artifact_type] = np.mean(artifact_scores)
            print(f"  Average {artifact_type} score: {results[artifact_type]:.3f}")
    
    # Summary
    print(f"\n--- SMOOTHING DETECTOR SUMMARY ---")
    for key, value in results.items():
        print(f"{key}: {value:.3f}")
    
    # Validate expected behavior
    if results.get('high_pixel', 0) > results.get('real', 1):
        print("✓ Detection working correctly: high_pixel samples show more smoothing than real faces")
    else:
        print("⚠ Detection may need tuning: high_pixel samples should show more smoothing")

if __name__ == "__main__":
    test_detector_on_dataset()
