import cv2
import numpy as np
from typing import Tuple


class CompressionEstimator:
    """Estimates video/image compression level from visual artifacts.

    Compression (JPEG, H.264/265) introduces blockiness, quantization noise,
    high-frequency energy loss, and posterization. These overlap with
    AI-generation artifact signatures, causing false positives. This estimator
    provides an attenuation factor that the classifier uses to down-weight
    compression-sensitive detector scores.

    This is a utility, not a detector -- it does not classify images.

    Parameters
    ----------
    weights : dict, optional
        Override default feature weights.
    """

    DEFAULT_WEIGHTS = {
        'blockiness': 0.35,
        'quantization': 0.25,
        'hf_loss': 0.25,
        'flat_region': 0.15,
    }

    def __init__(self, weights: dict | None = None) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def estimate_compression(self, image: np.ndarray) -> Tuple[float, dict]:
        """Estimate the compression level of an image.

        Parameters
        ----------
        image : np.ndarray
            Image (H, W, 3) in RGB colour space, or grayscale (H, W).

        Returns
        -------
        compression_level : float
            Overall compression estimate in [0, 1]. Higher = more compressed.
        details : dict
            Per-feature breakdown.
        """
        if image is None or image.size == 0:
            return 0.0, {
                'blockiness': 0.0,
                'quantization': 0.0,
                'hf_loss': 0.0,
                'flat_region': 0.0,
            }

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.0, {
                'blockiness': 0.0,
                'quantization': 0.0,
                'hf_loss': 0.0,
                'flat_region': 0.0,
            }

        blockiness = self._measure_blockiness(gray)
        quantization = self._measure_quantization(gray)
        hf_loss = self._measure_hf_loss(gray)
        flat_region = self._measure_flat_region(gray)

        compression_level = (
            self.weights['blockiness'] * blockiness
            + self.weights['quantization'] * quantization
            + self.weights['hf_loss'] * hf_loss
            + self.weights['flat_region'] * flat_region
        )
        compression_level = float(np.clip(compression_level, 0.0, 1.0))

        details = {
            'blockiness': blockiness,
            'quantization': quantization,
            'hf_loss': hf_loss,
            'flat_region': flat_region,
        }
        return compression_level, details

    def get_attenuation_factor(self, compression_level: float) -> float:
        """Convert compression level to a multiplicative attenuation factor.

        Parameters
        ----------
        compression_level : float
            Compression estimate in [0, 1].

        Returns
        -------
        float
            Attenuation factor in [0.3, 1.0].
            - compression < 0.2  -> 1.0 (no attenuation)
            - compression 0.2-0.8 -> linear decrease to 0.3
            - compression > 0.8  -> 0.3 (floor)
        """
        if compression_level <= 0.2:
            return 1.0
        if compression_level >= 0.8:
            return 0.3
        # Linear interpolation: 0.2 -> 1.0, 0.8 -> 0.3
        return 1.0 - (compression_level - 0.2) * (0.7 / 0.6)

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def _measure_blockiness(self, gray: np.ndarray) -> float:
        """Detect DCT block boundary artifacts (8x8 and 16x16).

        Compression codecs partition frames into 8x8 or 16x16 blocks for
        DCT. At low quality, block boundaries become visible as intensity
        discontinuities. We measure the ratio of pixel differences at
        block boundaries vs. interior positions.
        """
        gray_f = gray.astype(np.float64)
        h, w = gray_f.shape

        scores = []
        for block_size in (8, 16):
            if h < block_size * 2 or w < block_size * 2:
                continue

            # Horizontal block boundaries
            boundary_diffs = []
            interior_diffs = []

            for col in range(1, w):
                diff = np.mean(np.abs(gray_f[:, col] - gray_f[:, col - 1]))
                if col % block_size == 0:
                    boundary_diffs.append(diff)
                else:
                    interior_diffs.append(diff)

            # Vertical block boundaries
            for row in range(1, h):
                diff = np.mean(np.abs(gray_f[row, :] - gray_f[row - 1, :]))
                if row % block_size == 0:
                    boundary_diffs.append(diff)
                else:
                    interior_diffs.append(diff)

            if not boundary_diffs or not interior_diffs:
                continue

            avg_boundary = np.mean(boundary_diffs)
            avg_interior = np.mean(interior_diffs)

            # Ratio > 1 means boundaries are sharper than interior
            ratio = avg_boundary / (avg_interior + 1e-10)
            # Normalize: ratio of ~1.0 = no blockiness, ~1.5+ = heavy
            block_score = np.clip((ratio - 1.0) / 0.5, 0.0, 1.0)
            scores.append(block_score)

        if not scores:
            return 0.0
        return float(max(scores))

    def _measure_quantization(self, gray: np.ndarray) -> float:
        """Detect quantization step patterns in the noise histogram.

        Compression quantizes DCT coefficients, producing periodic peaks
        in the pixel-difference histogram. We detect these via
        autocorrelation of the noise histogram.
        """
        gray_f = gray.astype(np.float64)

        # High-pass filter to isolate noise
        blurred = cv2.GaussianBlur(gray_f, (5, 5), 1.0)
        noise = gray_f - blurred

        # Histogram of noise values
        noise_clipped = np.clip(noise, -50, 50)
        hist, _ = np.histogram(noise_clipped.flatten(), bins=101, range=(-50, 50))
        hist = hist.astype(np.float64)

        if np.max(hist) < 1:
            return 0.0

        # Normalize histogram
        hist = hist / (np.max(hist) + 1e-10)

        # Autocorrelation to find periodic peaks
        hist_centered = hist - np.mean(hist)
        autocorr = np.correlate(hist_centered, hist_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # positive lags only

        if len(autocorr) < 4 or autocorr[0] < 1e-10:
            return 0.0

        autocorr = autocorr / (autocorr[0] + 1e-10)  # normalize

        # Look for secondary peaks (lag > 2) indicating periodicity
        peaks = []
        for i in range(3, min(len(autocorr) - 1, 30)):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                peaks.append(autocorr[i])

        if not peaks:
            return 0.0

        max_peak = max(peaks)
        # Normalize: strong secondary peak = heavy quantization
        score = np.clip(max_peak / 0.3, 0.0, 1.0)
        return float(score)

    def _measure_hf_loss(self, gray: np.ndarray) -> float:
        """Measure high-frequency energy loss from compression.

        Compression discards high-frequency DCT coefficients first.
        We measure the ratio of high-frequency energy (outer FFT ring)
        to total energy. Uncompressed images typically have ratio > 0.15;
        heavily compressed images drop below 0.05.
        """
        gray_f = gray.astype(np.float64)
        f_transform = np.fft.fft2(gray_f)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        if max_r < 4:
            return 0.0

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # High-frequency: outer 25% of the spectrum
        hf_mask = r > (max_r * 0.75)
        total_energy = np.sum(magnitude) + 1e-10
        hf_energy = np.sum(magnitude[hf_mask])

        hf_ratio = hf_energy / total_energy

        # Map: ratio > 0.15 → low compression (score 0),
        #       ratio < 0.03 → heavy compression (score 1)
        score = 1.0 - np.clip((hf_ratio - 0.03) / 0.12, 0.0, 1.0)
        return float(score)

    def _measure_flat_region(self, gray: np.ndarray) -> float:
        """Detect posterization / banding in flat regions.

        Heavy compression reduces the number of distinct pixel values
        in smooth gradient areas. We find flat patches and count
        unique intensity values.
        """
        h, w = gray.shape
        patch_size = max(min(h, w) // 8, 8)

        unique_counts = []
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = gray[i:i + patch_size, j:j + patch_size]

                # Only consider "flat" patches (low variance)
                if np.std(patch) < 20:
                    unique_vals = len(np.unique(patch))
                    max_possible = min(patch_size * patch_size, 256)
                    unique_ratio = unique_vals / (max_possible + 1e-10)
                    unique_counts.append(unique_ratio)

        if not unique_counts:
            return 0.0

        avg_unique_ratio = np.mean(unique_counts)

        # Low unique ratio in flat patches = posterization from compression
        # Typical: uncompressed > 0.3, heavy compression < 0.1
        score = 1.0 - np.clip(avg_unique_ratio / 0.3, 0.0, 1.0)
        return float(score)
