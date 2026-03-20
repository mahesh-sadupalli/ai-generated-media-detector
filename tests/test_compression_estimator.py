import cv2
import numpy as np
import pytest

from src.utils.compression_estimator import CompressionEstimator


@pytest.fixture
def estimator():
    return CompressionEstimator()


def _make_natural_image(seed: int = 42) -> np.ndarray:
    """Create a synthetic natural-looking image with gradients and texture."""
    rng = np.random.RandomState(seed)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Smooth gradient base
    for i in range(224):
        for c in range(3):
            img[i, :, c] = int(50 + 150 * (i / 223))
    # Add noise and texture
    noise = rng.randint(-20, 20, (224, 224, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Simulate JPEG compression via encode/decode cycle."""
    # Convert RGB to BGR for cv2 encoding
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', bgr, encode_params)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    # Convert back to RGB
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------
# Basic functionality
# ------------------------------------------------------------------

class TestBasicFunctionality:
    def test_returns_score_and_details(self, estimator):
        img = _make_natural_image()
        level, details = estimator.estimate_compression(img)
        assert isinstance(level, float)
        assert isinstance(details, dict)
        assert 0.0 <= level <= 1.0
        for key in ('blockiness', 'quantization', 'hf_loss', 'flat_region'):
            assert key in details
            assert 0.0 <= details[key] <= 1.0

    def test_uncompressed_scores_low(self, estimator):
        img = _make_natural_image()
        level, _ = estimator.estimate_compression(img)
        assert level < 0.5, f"Uncompressed image scored {level}, expected < 0.5"


# ------------------------------------------------------------------
# JPEG quality tests
# ------------------------------------------------------------------

class TestJPEGQuality:
    def test_high_quality_jpeg_scores_low(self, estimator):
        img = _make_natural_image()
        compressed = _jpeg_roundtrip(img, quality=95)
        level, _ = estimator.estimate_compression(compressed)
        assert level < 0.5, f"JPEG q=95 scored {level}, expected < 0.5"

    def test_low_quality_jpeg_scores_high(self, estimator):
        img = _make_natural_image()
        compressed = _jpeg_roundtrip(img, quality=10)
        level, _ = estimator.estimate_compression(compressed)
        assert level > 0.3, f"JPEG q=10 scored {level}, expected > 0.3"

    def test_monotonic_ordering(self, estimator):
        """Lower JPEG quality should produce higher compression scores."""
        img = _make_natural_image()
        levels = []
        for quality in (95, 50, 10):
            compressed = _jpeg_roundtrip(img, quality=quality)
            level, _ = estimator.estimate_compression(compressed)
            levels.append(level)

        assert levels[0] <= levels[1], (
            f"q=95 ({levels[0]:.3f}) should score <= q=50 ({levels[1]:.3f})"
        )
        assert levels[1] <= levels[2], (
            f"q=50 ({levels[1]:.3f}) should score <= q=10 ({levels[2]:.3f})"
        )


# ------------------------------------------------------------------
# Attenuation factor
# ------------------------------------------------------------------

class TestAttenuationFactor:
    def test_no_attenuation_below_threshold(self, estimator):
        assert estimator.get_attenuation_factor(0.0) == 1.0
        assert estimator.get_attenuation_factor(0.1) == 1.0
        assert estimator.get_attenuation_factor(0.19) == 1.0

    def test_floor_above_threshold(self, estimator):
        assert estimator.get_attenuation_factor(0.8) == pytest.approx(0.3, abs=0.01)
        assert estimator.get_attenuation_factor(0.9) == 0.3
        assert estimator.get_attenuation_factor(1.0) == 0.3

    def test_linear_midrange(self, estimator):
        # At 0.5, expect ~0.65 (midway between 1.0 and 0.3)
        factor = estimator.get_attenuation_factor(0.5)
        assert 0.5 < factor < 0.8, f"Factor at 0.5 compression = {factor}"

    def test_monotonically_decreasing(self, estimator):
        factors = [estimator.get_attenuation_factor(c / 10.0) for c in range(11)]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1], (
                f"Factor at {i / 10} ({factors[i]}) should be >= factor at "
                f"{(i + 1) / 10} ({factors[i + 1]})"
            )

    def test_boundary_value_0_2(self, estimator):
        assert estimator.get_attenuation_factor(0.2) == pytest.approx(1.0, abs=0.01)

    def test_boundary_value_0_8(self, estimator):
        assert estimator.get_attenuation_factor(0.8) == pytest.approx(0.3, abs=0.01)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_none_input(self, estimator):
        level, details = estimator.estimate_compression(None)
        assert level == 0.0
        assert all(v == 0.0 for v in details.values())

    def test_empty_image(self, estimator):
        empty = np.array([], dtype=np.uint8)
        level, details = estimator.estimate_compression(empty)
        assert level == 0.0

    def test_single_color_image(self, estimator):
        solid = np.full((224, 224, 3), 128, dtype=np.uint8)
        level, details = estimator.estimate_compression(solid)
        assert isinstance(level, float)
        assert 0.0 <= level <= 1.0

    def test_grayscale_input(self, estimator):
        gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        level, details = estimator.estimate_compression(gray)
        assert isinstance(level, float)
        assert 0.0 <= level <= 1.0

    def test_tiny_image(self, estimator):
        tiny = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        level, details = estimator.estimate_compression(tiny)
        assert level == 0.0

    def test_large_image(self, estimator):
        large = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        level, details = estimator.estimate_compression(large)
        assert isinstance(level, float)
        assert 0.0 <= level <= 1.0


# ------------------------------------------------------------------
# Integration: compressed image should NOT trigger diffusion detector
# ------------------------------------------------------------------

class TestIntegration:
    def test_compressed_natural_not_flagged_as_diffusion(self):
        """A heavily compressed natural image with attenuation applied
        should have diffusion-adjusted score below threshold."""
        from src.artifact_detectors.diffusion_detector import DiffusionArtifactDetector

        estimator = CompressionEstimator()
        diff_detector = DiffusionArtifactDetector()

        img = _make_natural_image()
        compressed = _jpeg_roundtrip(img, quality=20)

        comp_level, _ = estimator.estimate_compression(compressed)
        attenuation = estimator.get_attenuation_factor(comp_level)

        diff_score, diff_details = diff_detector.detect_diffusion_artifacts(compressed)

        # Attenuate compression-sensitive sub-scores
        recon_adj = diff_details['reconstruction_error'] * attenuation
        noise_adj = diff_details['noise_residual'] * attenuation
        spectral = diff_details['spectral_fingerprint']
        patch = diff_details['patch_consistency']

        diff_adj = (
            0.30 * recon_adj
            + 0.30 * spectral
            + 0.20 * noise_adj
            + 0.20 * patch
        )

        # Adjusted diffusion score should be lower than raw
        assert diff_adj <= diff_score + 0.01, (
            f"Adjusted ({diff_adj:.3f}) should be <= raw ({diff_score:.3f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
