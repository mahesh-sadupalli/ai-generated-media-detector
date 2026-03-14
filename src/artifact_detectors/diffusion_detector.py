import cv2
import numpy as np
from scipy import stats
from typing import Tuple


class DiffusionArtifactDetector:
    """Detects artifacts specific to diffusion model generated images.

    Diffusion models (Stable Diffusion, Midjourney, DALL-E, Flux) produce
    high-quality, diverse images but leave distinct artifact signatures that
    differ from GAN-generated content. This detector targets those signatures
    using DIRE-inspired reconstruction analysis, spectral fingerprinting,
    noise residual statistics, and local patch consistency metrics.

    Parameters
    ----------
    threshold : float
        Score above which an image is considered suspicious. Default 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def detect_diffusion_artifacts(self, image: np.ndarray) -> Tuple[float, dict]:
        """Detect diffusion model artifacts in a face image.

        Parameters
        ----------
        image : np.ndarray
            Face image, expected shape (H, W, 3) in RGB colour space.

        Returns
        -------
        score : float
            Diffusion artifact score in [0, 1]. Higher = more likely
            diffusion-generated.
        details : dict
            Per-metric breakdown and human-readable interpretation.
        """
        if image is None or image.size == 0:
            return 0.0, {
                'reconstruction_error': 0.0,
                'spectral_fingerprint': 0.0,
                'noise_residual': 0.0,
                'patch_consistency': 0.0,
                'interpretation': 'Invalid image input',
            }

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        reconstruction_error = self._analyze_reconstruction_error(gray)
        spectral_fingerprint = self._analyze_spectral_fingerprint(gray)
        noise_residual = self._analyze_noise_residual(gray)
        patch_consistency = self._analyze_patch_consistency(gray)

        # Weighted combination — reconstruction and spectral are strongest
        # signals for diffusion detection
        diffusion_score = (
            0.30 * reconstruction_error
            + 0.30 * spectral_fingerprint
            + 0.20 * noise_residual
            + 0.20 * patch_consistency
        )
        diffusion_score = float(np.clip(diffusion_score, 0.0, 1.0))

        details = {
            'reconstruction_error': reconstruction_error,
            'spectral_fingerprint': spectral_fingerprint,
            'noise_residual': noise_residual,
            'patch_consistency': patch_consistency,
            'interpretation': self._interpret_score(diffusion_score),
        }

        return diffusion_score, details

    # ------------------------------------------------------------------
    # Sub-detectors
    # ------------------------------------------------------------------

    def _analyze_reconstruction_error(self, gray: np.ndarray) -> float:
        """DIRE-inspired reconstruction error proxy.

        Real images change significantly when blurred and sharpened (lossy
        reconstruction).  Diffusion-generated images lie closer to the
        learned manifold, so a blur-then-sharpen round-trip alters them
        less.  We measure this discrepancy at multiple scales.
        """
        gray_f = gray.astype(np.float64)
        errors = []

        for sigma in (1.0, 2.0, 4.0):
            ksize = int(6 * sigma + 1) | 1  # ensure odd
            blurred = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)

            # Sharpen via unsharp mask
            sharpened = cv2.addWeighted(gray_f, 1.5, blurred, -0.5, 0)

            # Reconstruction error: how much did the round-trip change?
            mse = np.mean((gray_f - sharpened) ** 2)
            errors.append(mse)

        avg_error = np.mean(errors)

        # Low reconstruction error → image is "smooth" in a way consistent
        # with lying on the diffusion manifold → higher suspicion score.
        # Normalise: empirically, real faces have errors ~200-800,
        # diffusion-generated ~50-200.
        score = 1.0 - min(avg_error / 500.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_spectral_fingerprint(self, gray: np.ndarray) -> float:
        """Detect diffusion-specific frequency domain signatures.

        GANs produce periodic spectral peaks (from upsampling layers).
        Diffusion models produce a more uniform spectrum but with a
        characteristic mid-frequency energy dip caused by the U-Net
        architecture's bottleneck.  We measure the ratio of mid-frequency
        energy to the expected smooth fall-off.
        """
        f_transform = np.fft.fft2(gray.astype(np.float64))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift) + 1e-10

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        # Radial average of the power spectrum
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        radial_bins = max_r
        radial_profile = np.zeros(radial_bins)
        counts = np.zeros(radial_bins)

        valid = r < radial_bins
        np.add.at(radial_profile, r[valid], magnitude[valid])
        np.add.at(counts, r[valid], 1)
        radial_profile /= (counts + 1e-10)

        if radial_bins < 10:
            return 0.0

        # Divide spectrum into low / mid / high bands
        third = radial_bins // 3
        low_energy = np.mean(radial_profile[:third])
        mid_energy = np.mean(radial_profile[third : 2 * third])
        high_energy = np.mean(radial_profile[2 * third :])

        # Diffusion models: mid-frequency dip relative to smooth fall-off
        # Expected ratio for natural images: mid ≈ geometric mean of low & high
        expected_mid = np.sqrt(low_energy * high_energy + 1e-10)
        mid_deviation = abs(mid_energy - expected_mid) / (expected_mid + 1e-10)

        # Also check spectral flatness (diffusion output is more uniform)
        log_profile = np.log(radial_profile + 1e-10)
        geometric_mean = np.exp(np.mean(log_profile))
        arithmetic_mean = np.mean(radial_profile)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

        # Combine: high flatness + mid-freq deviation → diffusion signature
        score = 0.5 * min(mid_deviation / 2.0, 1.0) + 0.5 * spectral_flatness
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_noise_residual(self, gray: np.ndarray) -> float:
        """Analyse noise residual statistics.

        Extract the noise layer by subtracting a denoised version.  Real
        camera noise follows sensor-specific distributions.  Diffusion
        model noise residuals tend to be more Gaussian and spatially
        uniform (a remnant of the denoising process).
        """
        gray_f = gray.astype(np.float64)

        # Denoise with non-local means (good at preserving edges)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, searchWindowSize=21)
        noise = gray_f - denoised.astype(np.float64)

        if np.std(noise) < 1e-10:
            return 0.0

        # 1. Gaussianity test (Jarque-Bera on sampled pixels)
        flat_noise = noise.flatten()
        sample_size = min(5000, len(flat_noise))
        rng = np.random.RandomState(42)
        sample = rng.choice(flat_noise, sample_size, replace=False)
        jb_stat, jb_p = stats.jarque_bera(sample)

        # High p-value → noise looks Gaussian → diffusion signature
        gaussianity = min(jb_p * 5.0, 1.0)  # scale up since p often < 0.2

        # 2. Spatial uniformity of noise variance
        patch_size = max(gray.shape[0] // 4, 8)
        variances = []
        for i in range(0, gray.shape[0] - patch_size + 1, patch_size):
            for j in range(0, gray.shape[1] - patch_size + 1, patch_size):
                patch_noise = noise[i : i + patch_size, j : j + patch_size]
                variances.append(np.var(patch_noise))

        if len(variances) < 2:
            uniformity = 0.5
        else:
            cv_variance = np.std(variances) / (np.mean(variances) + 1e-10)
            # Low coefficient of variation → spatially uniform noise → suspicious
            uniformity = 1.0 - min(cv_variance / 1.5, 1.0)

        # 3. Kurtosis — diffusion noise is closer to mesokurtic (kurtosis ≈ 0)
        kurt = abs(stats.kurtosis(flat_noise))
        kurtosis_score = 1.0 - min(kurt / 6.0, 1.0)

        score = 0.35 * gaussianity + 0.35 * uniformity + 0.30 * kurtosis_score
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_patch_consistency(self, gray: np.ndarray) -> float:
        """Detect patch-level inconsistencies from the denoising process.

        Diffusion models apply denoising through a U-Net that processes
        the image in local receptive fields.  This can create subtle
        boundaries between patches where texture statistics shift.  We
        measure the variance of per-patch texture features across the
        image.
        """
        h, w = gray.shape
        patch_size = max(h // 4, 16)
        gray_f = gray.astype(np.float64)

        patch_features = []
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = gray_f[i : i + patch_size, j : j + patch_size]

                # Compute local features per patch
                local_var = np.var(patch)
                grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                edge_density = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))

                # Frequency content of patch
                f = np.fft.fft2(patch)
                hf_ratio = np.sum(np.abs(f[patch_size // 4 :, :])) / (
                    np.sum(np.abs(f)) + 1e-10
                )

                patch_features.append([local_var, edge_density, hf_ratio])

        if len(patch_features) < 4:
            return 0.0

        features = np.array(patch_features)

        # Measure cross-patch consistency via coefficient of variation
        cvs = []
        for col in range(features.shape[1]):
            col_data = features[:, col]
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            cvs.append(std_val / (mean_val + 1e-10))

        avg_cv = np.mean(cvs)

        # Diffusion images: patches are more internally consistent but show
        # abrupt transitions at boundaries → moderate CV.
        # Real images: gradual natural variation → higher CV.
        # Very low CV (uniform) or very high CV (strong edges) are less suspicious.
        # Peak suspicion around CV ≈ 0.3-0.6.
        score = 1.0 - abs(avg_cv - 0.45) / 0.45
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation."""
        if score > 0.7:
            return "High diffusion artifacts detected — likely diffusion-generated"
        elif score > 0.4:
            return "Moderate diffusion signatures — suspicious"
        else:
            return "Low diffusion artifacts — appears natural or GAN-generated"
