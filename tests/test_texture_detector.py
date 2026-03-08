import cv2
import numpy as np
import os
from pathlib import Path

from src.artifact_detectors.texture_detector import TextureArtifactDetector
from src.utils.simple_face_detection import SimpleFaceDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_texture_detector_on_dataset():
    """Test texture detector on real vs generated samples"""
    detector = TextureArtifactDetector()
    face_extractor = SimpleFaceDetector()

    results = {}

    # Test on real faces
    print("Testing TEXTURE DETECTOR on REAL faces:")
    real_video = DATA_DIR / "raw" / "faceforensics" / "original_sequences" / "youtube" / "c23" / "videos" / "183.mp4"
    if real_video.exists():
        real_faces = face_extractor.extract_faces_from_video(str(real_video), max_frames=3)
        real_scores = []

        for i, face in enumerate(real_faces):
            score, details = detector.detect_texture_artifacts(face)
            real_scores.append(score)
            print(f"  Real face {i}: {score:.3f} - {details['interpretation']}")

        results['real'] = np.mean(real_scores)
        print(f"  Average real face score: {results['real']:.3f}")

    # Test on high perceptual samples (should have high texture artifact scores)
    print("\nTesting on HIGH_PERCEPTUAL artifacts (should detect texture issues):")
    perceptual_dir = DATA_DIR / "generated" / "high_perceptual"
    if perceptual_dir.exists():
        perceptual_scores = []
        perceptual_files = sorted(f for f in os.listdir(perceptual_dir) if f.endswith('.jpg'))[:5]

        for filename in perceptual_files:
            img_path = perceptual_dir / filename
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            score, details = detector.detect_texture_artifacts(img)
            perceptual_scores.append(score)
            print(f"  {filename}: {score:.3f} - {details['interpretation']}")

        results['high_perceptual'] = np.mean(perceptual_scores)
        print(f"  Average high_perceptual score: {results['high_perceptual']:.3f}")

    # Test on other artifact types
    for artifact_type in ['high_pixel', 'high_adversarial']:
        print(f"\nTesting on {artifact_type.upper()} artifacts:")
        artifact_dir = DATA_DIR / "generated" / artifact_type
        if artifact_dir.exists():
            artifact_scores = []
            artifact_files = sorted(f for f in os.listdir(artifact_dir) if f.endswith('.jpg'))[:3]

            for filename in artifact_files:
                img_path = artifact_dir / filename
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                score, details = detector.detect_texture_artifacts(img)
                artifact_scores.append(score)
                print(f"  {filename}: {score:.3f}")

            results[artifact_type] = np.mean(artifact_scores)
            print(f"  Average {artifact_type} score: {results[artifact_type]:.3f}")

    # Summary
    print(f"\n--- TEXTURE DETECTOR SUMMARY ---")
    for key, value in results.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    test_texture_detector_on_dataset()
