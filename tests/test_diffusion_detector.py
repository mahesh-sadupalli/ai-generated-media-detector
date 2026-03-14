import cv2
import numpy as np
import os
from pathlib import Path

from src.artifact_detectors.diffusion_detector import DiffusionArtifactDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_diffusion_detector_on_synthetic():
    """Test diffusion detector on synthetic images with known properties."""
    detector = DiffusionArtifactDetector()

    print("=== DIFFUSION ARTIFACT DETECTOR TESTS ===\n")

    # --- Test 1: Random noise image (should score low) ---
    print("Test 1: Random noise image")
    noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    score, details = detector.detect_diffusion_artifacts(noise_img)
    print(f"  Score: {score:.3f}")
    print(f"  Interpretation: {details['interpretation']}")
    print()

    # --- Test 2: Smooth gradient image (mimics diffusion over-smoothing) ---
    print("Test 2: Smooth gradient (diffusion-like)")
    gradient = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        gradient[i, :, :] = int(255 * i / 223)
    score_smooth, details_smooth = detector.detect_diffusion_artifacts(gradient)
    print(f"  Score: {score_smooth:.3f}")
    print(f"  Reconstruction error: {details_smooth['reconstruction_error']:.3f}")
    print(f"  Noise residual: {details_smooth['noise_residual']:.3f}")
    print()

    # --- Test 3: Natural-looking texture (should score low) ---
    print("Test 3: Textured image (natural-like)")
    rng = np.random.RandomState(123)
    base = rng.randint(80, 180, (224, 224, 3), dtype=np.uint8)
    # Add structured texture
    for i in range(224):
        for j in range(224):
            base[i, j, 0] = np.clip(base[i, j, 0] + int(20 * np.sin(i / 5.0)), 0, 255)
            base[i, j, 1] = np.clip(base[i, j, 1] + int(20 * np.cos(j / 7.0)), 0, 255)
    score_tex, details_tex = detector.detect_diffusion_artifacts(base)
    print(f"  Score: {score_tex:.3f}")
    print(f"  Spectral fingerprint: {details_tex['spectral_fingerprint']:.3f}")
    print()

    # --- Test 4: Edge case — None input ---
    print("Test 4: None input")
    score_none, details_none = detector.detect_diffusion_artifacts(None)
    print(f"  Score: {score_none:.3f}")
    print(f"  Interpretation: {details_none['interpretation']}")
    assert score_none == 0.0, "None input should return 0.0"
    print()

    # --- Test 5: Edge case — empty image ---
    print("Test 5: Empty image")
    empty = np.array([], dtype=np.uint8)
    score_empty, details_empty = detector.detect_diffusion_artifacts(empty)
    print(f"  Score: {score_empty:.3f}")
    assert score_empty == 0.0, "Empty input should return 0.0"
    print()

    # --- Test 6: Single-color image ---
    print("Test 6: Single-color image")
    solid = np.full((224, 224, 3), 128, dtype=np.uint8)
    score_solid, details_solid = detector.detect_diffusion_artifacts(solid)
    print(f"  Score: {score_solid:.3f}")
    print(f"  Interpretation: {details_solid['interpretation']}")
    print()

    # --- Test 7: Grayscale input ---
    print("Test 7: Grayscale input")
    gray_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    score_gray, details_gray = detector.detect_diffusion_artifacts(gray_img)
    print(f"  Score: {score_gray:.3f}")
    print()

    print("=== ALL TESTS PASSED ===")


def test_diffusion_detector_on_generated_samples():
    """Test diffusion detector on existing generated dataset samples."""
    detector = DiffusionArtifactDetector()

    print("\n=== DIFFUSION DETECTOR ON GENERATED SAMPLES ===\n")

    for artifact_type in ['high_pixel', 'high_perceptual', 'high_adversarial']:
        artifact_dir = DATA_DIR / "generated" / artifact_type
        if not artifact_dir.exists():
            print(f"  Skipping {artifact_type} (directory not found)")
            continue

        scores = []
        files = sorted(f for f in os.listdir(artifact_dir) if f.endswith('.jpg'))[:5]
        for filename in files:
            img = cv2.imread(str(artifact_dir / filename))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            score, _ = detector.detect_diffusion_artifacts(img)
            scores.append(score)

        if scores:
            print(f"  {artifact_type}: avg={np.mean(scores):.3f}  "
                  f"min={np.min(scores):.3f}  max={np.max(scores):.3f}")

    print()


if __name__ == "__main__":
    test_diffusion_detector_on_synthetic()
    test_diffusion_detector_on_generated_samples()
