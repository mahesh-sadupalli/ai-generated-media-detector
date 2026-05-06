import cv2
import numpy as np
import os
from pathlib import Path

from src.artifact_detectors.flux_detector import FluxArtifactDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_flux_detector_on_synthetic():
    """Test Flux detector on synthetic images with known properties."""
    detector = FluxArtifactDetector()

    print("=== FLUX ARTIFACT DETECTOR TESTS ===\n")

    # --- Test 1: Random noise image (should score low) ---
    print("Test 1: Random noise image")
    noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    score, details = detector.detect_flux_artifacts(noise_img)
    print(f"  Score: {score:.3f}")
    print(f"  Interpretation: {details['interpretation']}")
    assert 0.0 <= score <= 1.0, "Score must be in [0, 1]"
    print()

    # --- Test 2: Uniform smooth image (mimics global attention uniformity) ---
    print("Test 2: Uniform smooth image (Flux-like attention uniformity)")
    smooth = np.full((224, 224, 3), 128, dtype=np.uint8)
    # Add subtle per-channel variation (simulates GroupNorm decorrelation)
    smooth[:, :, 0] = 130
    smooth[:, :, 1] = 125
    smooth[:, :, 2] = 128
    # Add very mild Gaussian noise
    rng = np.random.RandomState(42)
    smooth = np.clip(
        smooth.astype(np.int16) + rng.randint(-3, 4, smooth.shape, dtype=np.int16),
        0, 255,
    ).astype(np.uint8)
    score_smooth, details_smooth = detector.detect_flux_artifacts(smooth)
    print(f"  Score: {score_smooth:.3f}")
    print(f"  Attention uniformity: {details_smooth['attention_uniformity']:.3f}")
    print(f"  VAE fingerprint: {details_smooth['vae_fingerprint']:.3f}")
    print()

    # --- Test 3: Natural-looking textured image (should score lower) ---
    print("Test 3: Natural textured image")
    rng = np.random.RandomState(123)
    base = rng.randint(60, 200, (224, 224, 3), dtype=np.uint8)
    # Add structured texture with high inter-channel correlation
    for i in range(224):
        for j in range(224):
            val = int(20 * np.sin(i / 5.0) + 15 * np.cos(j / 7.0))
            base[i, j, 0] = np.clip(base[i, j, 0] + val, 0, 255)
            base[i, j, 1] = np.clip(base[i, j, 1] + val, 0, 255)
            base[i, j, 2] = np.clip(base[i, j, 2] + val, 0, 255)
    score_tex, details_tex = detector.detect_flux_artifacts(base)
    print(f"  Score: {score_tex:.3f}")
    print(f"  Flow matching: {details_tex['flow_matching_residual']:.3f}")
    print()

    # --- Test 4: Simulated Flux-like image ---
    # Low inter-channel correlation, smooth edges, uniform patches
    print("Test 4: Simulated Flux-like characteristics")
    flux_sim = np.zeros((224, 224, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    # Each channel generated independently (decorrelated, like GroupNorm)
    flux_sim[:, :, 0] = np.clip(128 + rng.randn(224, 224) * 20, 0, 255).astype(np.uint8)
    flux_sim[:, :, 1] = np.clip(128 + rng.randn(224, 224) * 20, 0, 255).astype(np.uint8)
    flux_sim[:, :, 2] = np.clip(128 + rng.randn(224, 224) * 20, 0, 255).astype(np.uint8)
    # Apply mild smoothing (simulates no-skip-connection detail loss)
    flux_sim = cv2.GaussianBlur(flux_sim, (5, 5), 1.0)
    score_flux, details_flux = detector.detect_flux_artifacts(flux_sim)
    print(f"  Score: {score_flux:.3f}")
    print(f"  VAE fingerprint: {details_flux['vae_fingerprint']:.3f}")
    print(f"  Skip absence: {details_flux['skip_absence']:.3f}")
    print(f"  Attention uniformity: {details_flux['attention_uniformity']:.3f}")
    print()

    # --- Test 5: Edge case - None input ---
    print("Test 5: None input")
    score_none, details_none = detector.detect_flux_artifacts(None)
    print(f"  Score: {score_none:.3f}")
    assert score_none == 0.0, "None input should return 0.0"
    assert details_none['interpretation'] == 'Invalid image input'
    print()

    # --- Test 6: Edge case - empty image ---
    print("Test 6: Empty image")
    empty = np.array([], dtype=np.uint8)
    score_empty, _ = detector.detect_flux_artifacts(empty)
    assert score_empty == 0.0, "Empty input should return 0.0"
    print(f"  Score: {score_empty:.3f}")
    print()

    # --- Test 7: Grayscale input ---
    print("Test 7: Grayscale input")
    gray_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    score_gray, details_gray = detector.detect_flux_artifacts(gray_img)
    print(f"  Score: {score_gray:.3f}")
    assert 0.0 <= score_gray <= 1.0, "Score must be in [0, 1]"
    print()

    # --- Test 8: Small image ---
    print("Test 8: Small image (32x32)")
    small = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    score_small, _ = detector.detect_flux_artifacts(small)
    print(f"  Score: {score_small:.3f}")
    assert 0.0 <= score_small <= 1.0, "Score must be in [0, 1]"
    print()

    # --- Test 9: Verify all detail keys present ---
    print("Test 9: Verify output structure")
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    _, details = detector.detect_flux_artifacts(test_img)
    expected_keys = {
        'vae_fingerprint', 'flow_matching_residual', 'skip_absence',
        'distillation_artifacts', 'attention_uniformity', 'interpretation',
    }
    assert set(details.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(details.keys())}"
    )
    for key in expected_keys - {'interpretation'}:
        assert 0.0 <= details[key] <= 1.0, f"{key} out of range: {details[key]}"
    print("  All keys present and in valid range")
    print()

    # --- Test 10: Distillation artifact detection ---
    print("Test 10: Bimodal complexity (Schnell distillation-like)")
    dist_img = np.zeros((224, 224, 3), dtype=np.uint8)
    rng = np.random.RandomState(77)
    # Top half: highly detailed (sharp edges)
    dist_img[:112, :, :] = rng.randint(0, 255, (112, 224, 3), dtype=np.uint8)
    # Bottom half: very smooth (under-resolved)
    dist_img[112:, :, :] = 128
    score_dist, details_dist = detector.detect_flux_artifacts(dist_img)
    print(f"  Score: {score_dist:.3f}")
    print(f"  Distillation artifacts: {details_dist['distillation_artifacts']:.3f}")
    print()

    print("=== ALL TESTS PASSED ===")


def test_flux_detector_on_generated_samples():
    """Test Flux detector on existing generated dataset samples."""
    detector = FluxArtifactDetector()

    print("\n=== FLUX DETECTOR ON GENERATED SAMPLES ===\n")

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
            score, _ = detector.detect_flux_artifacts(img)
            scores.append(score)

        if scores:
            print(f"  {artifact_type}: avg={np.mean(scores):.3f}  "
                  f"min={np.min(scores):.3f}  max={np.max(scores):.3f}")

    print()


if __name__ == "__main__":
    test_flux_detector_on_synthetic()
    test_flux_detector_on_generated_samples()
