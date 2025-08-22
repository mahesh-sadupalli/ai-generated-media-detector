import cv2
import numpy as np
import os
import sys
sys.path.append('../utils')
from simple_face_detection import SimpleFaceDetector
from mode_collapse_detector import ModeCollapseDetector

def test_mode_collapse_detector_on_dataset():
    """Test mode collapse detector on real vs generated samples"""
    detector = ModeCollapseDetector()
    face_extractor = SimpleFaceDetector()
    
    results = {}
    
    # Test on real faces
    print("Testing MODE COLLAPSE DETECTOR on REAL faces:")
    real_video = "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/183.mp4"
    if os.path.exists(real_video):
        real_faces = face_extractor.extract_faces_from_video(real_video, max_frames=3)
        real_scores = []
        
        for i, face in enumerate(real_faces):
            score, details = detector.detect_mode_collapse_artifacts(face)
            real_scores.append(score)
            print(f"  Real face {i}: {score:.3f} - {details['interpretation']}")
        
        results['real'] = np.mean(real_scores)
        print(f"  Average real face score: {results['real']:.3f}")
    
    # Test on high adversarial samples (should have high mode collapse scores)
    print("\nTesting on HIGH_ADVERSARIAL artifacts (should detect mode collapse):")
    adversarial_dir = "../../data/generated/high_adversarial"
    if os.path.exists(adversarial_dir):
        adversarial_scores = []
        adversarial_files = [f for f in os.listdir(adversarial_dir) if f.endswith('.jpg')][:5]
        
        for filename in adversarial_files:
            img_path = os.path.join(adversarial_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            score, details = detector.detect_mode_collapse_artifacts(img)
            adversarial_scores.append(score)
            print(f"  {filename}: {score:.3f} - {details['interpretation']}")
        
        results['high_adversarial'] = np.mean(adversarial_scores)
        print(f"  Average high_adversarial score: {results['high_adversarial']:.3f}")
    
    # Test on other artifact types
    for artifact_type in ['high_pixel', 'high_perceptual']:
        print(f"\nTesting on {artifact_type.upper()} artifacts:")
        artifact_dir = f"../../data/generated/{artifact_type}"
        if os.path.exists(artifact_dir):
            artifact_scores = []
            artifact_files = [f for f in os.listdir(artifact_dir) if f.endswith('.jpg')][:3]
            
            for filename in artifact_files:
                img_path = os.path.join(artifact_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                score, details = detector.detect_mode_collapse_artifacts(img)
                artifact_scores.append(score)
                print(f"  {filename}: {score:.3f}")
            
            results[artifact_type] = np.mean(artifact_scores)
            print(f"  Average {artifact_type} score: {results[artifact_type]:.3f}")
    
    # Summary
    print(f"\n--- MODE COLLAPSE DETECTOR SUMMARY ---")
    for key, value in results.items():
        print(f"{key}: {value:.3f}")

if __name__ == "__main__":
    test_mode_collapse_detector_on_dataset()
