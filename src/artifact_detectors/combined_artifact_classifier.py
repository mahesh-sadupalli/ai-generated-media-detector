import cv2
import numpy as np
from typing import Tuple, Dict
import os
import sys
sys.path.append('../utils')
from simple_face_detection import SimpleFaceDetector
from smoothing_detector import SmoothingArtifactDetector
from texture_detector import TextureArtifactDetector
from mode_collapse_detector import ModeCollapseDetector

class GANArtifactClassifier:
    """Combined classifier using all three artifact detectors"""
    
    def __init__(self):
        self.smoothing_detector = SmoothingArtifactDetector()
        self.texture_detector = TextureArtifactDetector()
        self.mode_collapse_detector = ModeCollapseDetector()
        
        # Thresholds based on our testing
        self.thresholds = {
            'smoothing': 0.65,  # Above this suggests generated content
            'mode_collapse': 0.70,  # Above this suggests adversarial artifacts
            'fake_confidence': 0.60  # Overall fake confidence threshold
        }
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Comprehensive artifact analysis of an image
        
        Args:
            image: Face image (224x224x3)
            
        Returns:
            Dictionary with detailed analysis and final prediction
        """
        # Run all detectors
        smoothing_score, smoothing_details = self.smoothing_detector.detect_smoothing_artifacts(image)
        texture_score, texture_details = self.texture_detector.detect_texture_artifacts(image)
        collapse_score, collapse_details = self.mode_collapse_detector.detect_mode_collapse_artifacts(image)
        
        # Combine scores with weights based on our validation results
        # Smoothing and mode collapse showed better discrimination
        overall_score = (
            0.5 * smoothing_score +
            0.1 * texture_score +
            0.4 * collapse_score
        )
        
        # Determine prediction
        prediction = "FAKE" if overall_score > self.thresholds['fake_confidence'] else "REAL"
        confidence = overall_score
        
        # Identify dominant artifact type
        artifact_type = "unknown"
        if smoothing_score > self.thresholds['smoothing']:
            artifact_type = "pixel_loss_artifacts"
        elif collapse_score > self.thresholds['mode_collapse']:
            artifact_type = "adversarial_loss_artifacts"
        elif smoothing_score > 0.5 and collapse_score > 0.5:
            artifact_type = "mixed_artifacts"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'artifact_type': artifact_type,
            'scores': {
                'smoothing': smoothing_score,
                'texture': texture_score,
                'mode_collapse': collapse_score,
                'overall': overall_score
            },
            'details': {
                'smoothing': smoothing_details,
                'texture': texture_details,
                'mode_collapse': collapse_details
            },
            'explanation': self._generate_explanation(
                prediction, artifact_type, smoothing_score, texture_score, collapse_score
            )
        }
    
    def analyze_video(self, video_path: str, max_frames: int = 10) -> Dict:
        """
        Analyze video for deepfake artifacts
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze
            
        Returns:
            Dictionary with video-level analysis
        """
        face_extractor = SimpleFaceDetector()
        faces = face_extractor.extract_faces_from_video(video_path, max_frames=max_frames)
        
        if not faces:
            return {
                'prediction': "ERROR",
                'confidence': 0.0,
                'explanation': "No faces detected in video"
            }
        
        # Analyze each face
        frame_results = []
        for i, face in enumerate(faces):
            result = self.analyze_image(face)
            frame_results.append(result)
        
        # Aggregate results
        avg_confidence = np.mean([r['confidence'] for r in frame_results])
        fake_votes = sum(1 for r in frame_results if r['prediction'] == "FAKE")
        
        video_prediction = "FAKE" if fake_votes > len(frame_results) / 2 else "REAL"
        
        # Identify most common artifact type
        artifact_types = [r['artifact_type'] for r in frame_results if r['artifact_type'] != "unknown"]
        most_common_artifact = max(set(artifact_types), key=artifact_types.count) if artifact_types else "unknown"
        
        return {
            'prediction': video_prediction,
            'confidence': avg_confidence,
            'artifact_type': most_common_artifact,
            'frames_analyzed': len(faces),
            'fake_frames': fake_votes,
            'frame_results': frame_results,
            'explanation': f"Video analysis: {fake_votes}/{len(faces)} frames flagged as fake. "
                          f"Primary artifact type: {most_common_artifact}"
        }
    
    def _generate_explanation(self, prediction: str, artifact_type: str, 
                            smoothing: float, texture: float, collapse: float) -> str:
        """Generate human-readable explanation"""
        
        explanation = f"Prediction: {prediction} (confidence: {(smoothing*0.5 + texture*0.1 + collapse*0.4):.3f})\n"
        
        if prediction == "FAKE":
            explanation += "Detected issues:\n"
            if smoothing > 0.65:
                explanation += "- High smoothing artifacts suggest pixel-loss optimization\n"
            if collapse > 0.70:
                explanation += "- Mode collapse patterns suggest adversarial training issues\n"
            if texture > 0.50:
                explanation += "- Texture inconsistencies detected\n"
            
            explanation += f"\nDominant artifact type: {artifact_type}"
        else:
            explanation += "Image appears authentic based on artifact analysis."
        
        return explanation

def test_combined_classifier():
    """Test the combined classifier on our dataset"""
    classifier = GANArtifactClassifier()
    
    print("=== COMBINED GAN ARTIFACT CLASSIFIER TEST ===\n")
    
    # Test on real video
    real_video = "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/183.mp4"
    if os.path.exists(real_video):
        print("Testing on REAL video:")
        result = classifier.analyze_video(real_video, max_frames=3)
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Explanation: {result['explanation']}")
        print()
    
    # Test on deepfake video
    fake_video_dir = "../../data/raw/faceforensics/manipulated_sequences/Deepfakes/c23/videos"
    if os.path.exists(fake_video_dir):
        fake_files = [f for f in os.listdir(fake_video_dir) if f.endswith('.mp4')]
        if fake_files:
            fake_video = os.path.join(fake_video_dir, fake_files[0])
            print("Testing on DEEPFAKE video:")
            result = classifier.analyze_video(fake_video, max_frames=3)
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Explanation: {result['explanation']}")
            print()
    
    # Test on generated samples
    print("Testing on GENERATED SAMPLES:")
    for artifact_type in ['high_pixel', 'high_perceptual', 'high_adversarial']:
        artifact_dir = f"../../data/generated/{artifact_type}"
        if os.path.exists(artifact_dir):
            files = [f for f in os.listdir(artifact_dir) if f.endswith('.jpg')]
            if files:
                img_path = os.path.join(artifact_dir, files[0])
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                result = classifier.analyze_image(img)
                print(f"  {artifact_type}:")
                print(f"    Prediction: {result['prediction']}")
                print(f"    Confidence: {result['confidence']:.3f}")
                print(f"    Artifact type: {result['artifact_type']}")
                print()

if __name__ == "__main__":
    test_combined_classifier()
