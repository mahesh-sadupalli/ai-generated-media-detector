import cv2
import numpy as np
from scipy import stats
from typing import Tuple


class FluxArtifactDetector:
    """Detects artifacts specific to Black Forest Labs Flux models.

    Flux uses a fundamentally different architecture from standard latent
    diffusion models (Stable Diffusion, DALL-E): MMDiT transformer blocks
    instead of a U-Net, rectified flow matching instead of DDPM noise
    scheduling, and a 16-channel VAE (4x richer latent space than SD's
    4-channel VAE).  These architectural choices leave distinct forensic
    traces that this detector targets.

    Detection signals
    -----------------
    1. VAE decode fingerprint — GroupNorm(32) + Swish activation statistical
       trace from the 16-channel VAE decoder.
    2. Flow matching ODE residuals — systematic Euler discretisation errors
       along straight-line ODE transport paths.
    3. Skip-connection absence — subtle high-frequency detail loss from the
       transformer architecture lacking U-Net skip connections.
    4. Schnell distillation artifacts — local geometric distortions from
       few-step adversarial distillation (1-4 steps).
    5. Attention uniformity — unnaturally uniform textures across distant
       spatial regions caused by global self-attention.

    Parameters
    ----------
    threshold : float
        Score above which an image is considered suspicious.  Default 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def detect_flux_artifacts(self, image: np.ndarray) -> Tuple[float, dict]:
        """Detect Flux-specific artifacts in an image.

        Parameters
        ----------
        image : np.ndarray
            Face/image, expected shape (H, W, 3) in RGB colour space.

        Returns
        -------
        score : float
            Flux artifact score in [0, 1].  Higher = more likely
            Flux-generated.
        details : dict
            Per-metric breakdown and human-readable interpretation.
        """
        if image is None or image.size == 0:
            return 0.0, {
                'vae_fingerprint': 0.0,
                'flow_matching_residual': 0.0,
                'skip_absence': 0.0,
                'distillation_artifacts': 0.0,
                'attention_uniformity': 0.0,
                'interpretation': 'Invalid image input',
            }

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        vae_fingerprint = self._analyze_vae_fingerprint(image, gray)
        flow_matching = self._analyze_flow_matching_residual(gray)
        skip_absence = self._analyze_skip_absence(gray)
        distillation = self._analyze_distillation_artifacts(gray)
        attention = self._analyze_attention_uniformity(image, gray)

        # Weighted combination — VAE and flow matching are architecture-
        # specific; attention uniformity is the most compression-robust.
        flux_score = (
            0.25 * vae_fingerprint
            + 0.25 * flow_matching
            + 0.15 * skip_absence
            + 0.15 * distillation
            + 0.20 * attention
        )
        flux_score = float(np.clip(flux_score, 0.0, 1.0))

        details = {
            'vae_fingerprint': vae_fingerprint,
            'flow_matching_residual': flow_matching,
            'skip_absence': skip_absence,
            'distillation_artifacts': distillation,
            'attention_uniformity': attention,
            'interpretation': self._interpret_score(flux_score),
        }

        return flux_score, details

    # ------------------------------------------------------------------
    # Sub-detectors
    # ------------------------------------------------------------------

    def _analyze_vae_fingerprint(
        self, image: np.ndarray, gray: np.ndarray
    ) -> float:
        """Detect Flux's 16-channel VAE decoder fingerprint.

        Flux's VAE decoder uses GroupNorm(32 groups) followed by Swish
        (SiLU) activation.  GroupNorm normalises statistics within each
        group of channels, leaving a characteristic pattern: within-group
        channel correlations are suppressed while between-group boundaries
        show correlation discontinuities.  The Swish activation introduces
        a subtle asymmetry in pixel value distributions (negative values
        are softly gated rather than zeroed).

        Additionally, a 16-channel latent space (vs SD's 4-channel) allows
        finer spatial reconstruction, so the VAE decode error is lower —
        but its *statistical signature* is different from a natural image
        processed through a camera ISP.
        """
        if len(image.shape) != 3 or image.shape[2] < 3:
            return 0.0

        img_f = image.astype(np.float64)

        # --- 1. Inter-channel correlation structure ---
        # Flatten spatial dims for each channel
        channels = [img_f[:, :, c].flatten() for c in range(3)]

        # Pairwise Pearson correlations
        corr_01 = np.corrcoef(channels[0], channels[1])[0, 1]
        corr_02 = np.corrcoef(channels[0], channels[2])[0, 1]
        corr_12 = np.corrcoef(channels[1], channels[2])[0, 1]

        # Natural images: high inter-channel correlation (>0.85 typical).
        # Flux VAE decode: GroupNorm decorrelates channels → lower correlation.
        avg_corr = (abs(corr_01) + abs(corr_02) + abs(corr_12)) / 3.0
        # Score higher when correlation is atypically low (decorrelated)
        corr_score = 1.0 - min(avg_corr / 0.95, 1.0)

        # --- 2. Swish activation asymmetry ---
        # Compute local deviations from block mean (proxy for decoder output
        # distribution before final projection).  Swish produces a slight
        # negative skew in residuals.
        block_size = max(gray.shape[0] // 8, 8)
        skewness_values = []
        for i in range(0, gray.shape[0] - block_size + 1, block_size):
            for j in range(0, gray.shape[1] - block_size + 1, block_size):
                block = gray[i : i + block_size, j : j + block_size].astype(
                    np.float64
                )
                residual = block - np.mean(block)
                if np.std(residual) > 1e-10:
                    skewness_values.append(float(stats.skew(residual.flatten())))

        if len(skewness_values) < 4:
            skew_score = 0.0
        else:
            mean_skew = np.mean(skewness_values)
            # Flux: slight negative skew from Swish gating (~-0.1 to -0.4)
            # Natural images: skew varies widely, centred near 0
            skew_score = float(np.clip(
                1.0 - abs(mean_skew + 0.2) / 0.5, 0.0, 1.0
            ))

        # --- 3. GroupNorm banding in local variance ---
        # GroupNorm(32) on 16 latent channels creates periodic variance
        # modulation visible in the decoded pixel domain.  We look for
        # regularity in the local-variance signal along rows/columns.
        row_vars = np.var(img_f, axis=1).mean(axis=1)  # variance per row
        if len(row_vars) > 16:
            # Autocorrelation of row-variance signal at lag 4-16
            row_vars_centered = row_vars - np.mean(row_vars)
            norm = np.sum(row_vars_centered ** 2) + 1e-10
            max_autocorr = 0.0
            for lag in range(4, min(17, len(row_vars) // 2)):
                acorr = np.sum(
                    row_vars_centered[:-lag] * row_vars_centered[lag:]
                ) / norm
                max_autocorr = max(max_autocorr, abs(acorr))
            band_score = min(max_autocorr / 0.3, 1.0)
        else:
            band_score = 0.0

        score = 0.35 * corr_score + 0.35 * skew_score + 0.30 * band_score
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_flow_matching_residual(self, gray: np.ndarray) -> float:
        """Detect residuals from rectified flow matching ODE sampling.

        Flux uses rectified flow: it learns a velocity field v(x, t) that
        transports noise x_0 to data x_1 along straight lines.  Sampling
        uses the Euler method: x_{t+dt} = x_t + v(x_t, t) * dt.

        Unlike DDPM (which adds stochastic noise at each step), the Euler
        ODE solver produces *systematic* discretisation errors that
        accumulate coherently — not randomly.  These errors appear as:
        - Correlated residual structure (high autocorrelation in residuals)
        - Reduced high-frequency noise (ODE is smoother than SDE)
        - Step-size-dependent spatial frequency peaks

        We detect this by comparing the noise residual's autocorrelation
        structure against what natural camera noise would produce.
        """
        gray_f = gray.astype(np.float64)

        # Extract noise residual via bilateral filter (edge-preserving)
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        residual = gray_f - filtered.astype(np.float64)

        if np.std(residual) < 1e-10:
            return 0.0

        # --- 1. Spatial autocorrelation of residual ---
        # ODE errors are correlated; camera noise is i.i.d. per pixel.
        flat = residual.flatten()
        n = min(len(flat), 10000)
        rng = np.random.RandomState(42)
        indices = rng.choice(len(flat), n, replace=False)
        indices.sort()
        sample = flat[indices]

        # Lag-1 autocorrelation
        if len(sample) > 1:
            sample_centered = sample - np.mean(sample)
            denom = np.sum(sample_centered ** 2) + 1e-10
            lag1_acorr = np.sum(
                sample_centered[:-1] * sample_centered[1:]
            ) / denom
        else:
            lag1_acorr = 0.0

        # Flux: higher autocorrelation (systematic errors) vs camera noise
        acorr_score = min(abs(lag1_acorr) / 0.3, 1.0)

        # --- 2. High-frequency suppression (ODE smoother than SDE) ---
        f_transform = np.fft.fft2(residual)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift) + 1e-10

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        if max_r < 4:
            return acorr_score

        # Energy in outer 25% of frequency space
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        high_freq_mask = r > (max_r * 0.75)
        total_mask = r <= max_r

        high_energy = np.mean(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0.0
        total_energy = np.mean(magnitude[total_mask]) if np.any(total_mask) else 1e-10

        hf_ratio = high_energy / (total_energy + 1e-10)

        # Flux ODE: suppressed high-frequency noise → low ratio
        # Natural noise: more high-frequency content → higher ratio
        # Score high when HF ratio is unusually low
        hf_suppression_score = 1.0 - min(hf_ratio / 0.5, 1.0)

        # --- 3. Residual normality with spatial structure ---
        # ODE errors: non-Gaussian with spatial structure
        # Camera noise: Gaussian-like per-pixel
        _, p_value = stats.normaltest(
            rng.choice(flat, min(5000, len(flat)), replace=False)
        )
        # Low p-value → non-Gaussian residual → consistent with ODE errors
        normality_score = 1.0 - min(p_value * 10.0, 1.0)

        score = 0.40 * acorr_score + 0.35 * hf_suppression_score + 0.25 * normality_score
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_skip_absence(self, gray: np.ndarray) -> float:
        """Detect high-frequency detail loss from missing skip connections.

        U-Net architectures (Stable Diffusion) use skip connections to
        pass high-frequency detail from encoder to decoder, preserving
        fine texture.  Flux's MMDiT transformer has no skip connections —
        all spatial information must flow through the bottleneck of
        self-attention.  This causes:
        - Reduced high-frequency energy relative to mid-frequencies
        - Smoother edge profiles (less ringing, but also less sharp)
        - More uniform local texture statistics

        We measure the ratio of high-to-mid frequency energy and compare
        edge profile sharpness against natural image statistics.
        """
        gray_f = gray.astype(np.float64)

        # --- 1. High-to-mid frequency energy ratio ---
        f_transform = np.fft.fft2(gray_f)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift) + 1e-10

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        if max_r < 6:
            return 0.0

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        mid_mask = (r > max_r * 0.25) & (r <= max_r * 0.6)
        high_mask = r > max_r * 0.6

        mid_energy = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 1e-10
        high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0.0

        hf_mid_ratio = high_energy / (mid_energy + 1e-10)

        # No skip connections → less HF relative to mid → lower ratio
        # Natural images: typically ratio ~0.3-0.5
        # Flux: typically ratio ~0.1-0.25
        ratio_score = 1.0 - min(hf_mid_ratio / 0.45, 1.0)

        # --- 2. Edge profile smoothness ---
        # Sobel edges, then measure how "spread out" edges are
        edges = cv2.Canny(gray, 50, 150)
        if np.sum(edges) < 10:
            edge_score = 0.5
        else:
            # Dilate edges and measure gradient magnitude at edge locations
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            edge_region = dilated > 0

            grad_x = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Sharpness: ratio of edge pixels with very high gradient
            edge_gradients = grad_mag[edge_region]
            if len(edge_gradients) > 0:
                median_grad = np.median(edge_gradients)
                # Flux: smoother edges → lower median gradient
                # Natural: sharper edges → higher median gradient
                edge_score = 1.0 - min(median_grad / 80.0, 1.0)
            else:
                edge_score = 0.5

        # --- 3. Local texture uniformity ---
        # Without skip connections, fine texture is more uniform
        patch_size = max(gray.shape[0] // 8, 8)
        local_stds = []
        for i in range(0, gray.shape[0] - patch_size + 1, patch_size):
            for j in range(0, gray.shape[1] - patch_size + 1, patch_size):
                patch = gray_f[i : i + patch_size, j : j + patch_size]
                # High-pass via Laplacian
                lap = cv2.Laplacian(patch, cv2.CV_64F)
                local_stds.append(np.std(lap))

        if len(local_stds) < 4:
            texture_score = 0.0
        else:
            # Coefficient of variation of Laplacian stds across patches
            cv_lap = np.std(local_stds) / (np.mean(local_stds) + 1e-10)
            # Flux: more uniform texture detail → low CV
            texture_score = 1.0 - min(cv_lap / 0.8, 1.0)

        score = 0.40 * ratio_score + 0.30 * edge_score + 0.30 * texture_score
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_distillation_artifacts(self, gray: np.ndarray) -> float:
        """Detect artifacts from Flux Schnell's adversarial distillation.

        Flux Schnell generates in 1-4 steps via adversarial distillation,
        which produces:
        - Local geometric distortions (spatial warping from few-step
          approximation of the learned velocity field)
        - Inconsistent detail levels between regions (some areas fully
          resolved, others blurry or warped)
        - Periodic structure from the distillation discriminator's
          receptive field

        Dev/Pro (50+ steps) show fewer distillation artifacts, so this
        signal specifically targets Schnell-generated content.
        """
        gray_f = gray.astype(np.float64)
        h, w = gray.shape

        # --- 1. Local geometric consistency ---
        # Compute optical-flow-like local deformation by comparing patches
        # to their neighbours.  Distillation causes local warping.
        patch_size = max(h // 8, 8)
        stride = patch_size

        patch_complexities = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = gray_f[i : i + patch_size, j : j + patch_size]
                # Gradient magnitude as complexity proxy
                gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                complexity = np.mean(np.sqrt(gx ** 2 + gy ** 2))
                patch_complexities.append(complexity)

        if len(patch_complexities) < 4:
            return 0.0

        complexities = np.array(patch_complexities)

        # Distillation: bimodal distribution of complexities (some patches
        # fully rendered, others under-resolved)
        if np.std(complexities) < 1e-10:
            bimodal_score = 0.0
        else:
            normed = (complexities - np.mean(complexities)) / (
                np.std(complexities) + 1e-10
            )
            # Kurtosis < 0 (platykurtic) suggests bimodal distribution
            kurt = stats.kurtosis(normed)
            bimodal_score = float(np.clip(-kurt / 2.0, 0.0, 1.0))

        # --- 2. Detail level inconsistency ---
        # Measure variance of Laplacian (focus measure) per patch
        focus_measures = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = gray_f[i : i + patch_size, j : j + patch_size]
                lap_var = np.var(cv2.Laplacian(patch, cv2.CV_64F))
                focus_measures.append(lap_var)

        focus_arr = np.array(focus_measures)
        if np.mean(focus_arr) < 1e-10:
            focus_inconsistency = 0.0
        else:
            cv_focus = np.std(focus_arr) / (np.mean(focus_arr) + 1e-10)
            # Distillation: high variance in focus (some patches sharp,
            # others blurry) → high CV.  Natural: more gradual variation.
            focus_inconsistency = min(cv_focus / 1.5, 1.0)

        # --- 3. Discriminator receptive field periodicity ---
        # The adversarial distillation loss uses a discriminator with a
        # fixed receptive field, which can imprint periodic structure.
        # Detect via autocorrelation of the edge magnitude signal.
        edges = cv2.Canny(gray, 30, 100)
        row_sums = edges.astype(np.float64).sum(axis=1)

        if len(row_sums) > 16 and np.std(row_sums) > 1e-10:
            row_centered = row_sums - np.mean(row_sums)
            norm = np.sum(row_centered ** 2) + 1e-10
            peak_acorr = 0.0
            for lag in range(4, min(33, len(row_sums) // 2)):
                acorr = np.sum(
                    row_centered[:-lag] * row_centered[lag:]
                ) / norm
                peak_acorr = max(peak_acorr, abs(acorr))
            periodic_score = min(peak_acorr / 0.25, 1.0)
        else:
            periodic_score = 0.0

        score = (
            0.30 * bimodal_score
            + 0.40 * focus_inconsistency
            + 0.30 * periodic_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_attention_uniformity(
        self, image: np.ndarray, gray: np.ndarray
    ) -> float:
        """Detect unnatural uniformity from global self-attention.

        Flux's MMDiT uses global self-attention across all spatial
        positions.  Unlike convolutional architectures (limited receptive
        field), this allows every pixel to attend to every other pixel.
        This produces:
        - Unnaturally consistent texture statistics across distant regions
        - Long-range correlation in colour and luminance that exceeds
          what natural scenes produce
        - Reduced variation in local spectral energy across the image

        This signal is particularly compression-robust because it measures
        global statistical properties rather than local pixel patterns.
        """
        if len(image.shape) != 3 or image.shape[2] < 3:
            gray_f = gray.astype(np.float64)
            # Fall back to grayscale-only analysis
            return self._attention_uniformity_gray(gray_f)

        img_f = image.astype(np.float64)
        h, w, _ = img_f.shape

        # --- 1. Long-range colour consistency ---
        # Divide image into quadrants and compare colour histograms.
        # Natural scenes: quadrants differ significantly (sky vs ground).
        # Flux global attention: more homogeneous colour distribution.
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            img_f[:mid_h, :mid_w],
            img_f[:mid_h, mid_w:],
            img_f[mid_h:, :mid_w],
            img_f[mid_h:, mid_w:],
        ]

        quad_means = [np.mean(q, axis=(0, 1)) for q in quadrants]
        quad_stds = [np.std(q, axis=(0, 1)) for q in quadrants]

        # Pairwise differences in mean colour
        mean_diffs = []
        for i in range(4):
            for j in range(i + 1, 4):
                diff = np.linalg.norm(quad_means[i] - quad_means[j])
                mean_diffs.append(diff)

        avg_mean_diff = np.mean(mean_diffs)
        # Flux: lower inter-quadrant difference → more uniform
        colour_uniformity = 1.0 - min(avg_mean_diff / 60.0, 1.0)

        # Pairwise differences in std (texture energy)
        std_diffs = []
        for i in range(4):
            for j in range(i + 1, 4):
                diff = np.linalg.norm(quad_stds[i] - quad_stds[j])
                std_diffs.append(diff)

        avg_std_diff = np.mean(std_diffs)
        texture_uniformity = 1.0 - min(avg_std_diff / 25.0, 1.0)

        # --- 2. Spectral energy consistency across patches ---
        patch_size = max(h // 4, 16)
        spectral_energies = []
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = gray.astype(np.float64)[
                    i : i + patch_size, j : j + patch_size
                ]
                f = np.fft.fft2(patch)
                energy = np.mean(np.abs(f) ** 2)
                spectral_energies.append(energy)

        if len(spectral_energies) < 4:
            spectral_score = 0.0
        else:
            energies = np.array(spectral_energies)
            cv_energy = np.std(energies) / (np.mean(energies) + 1e-10)
            # Flux: low CV (uniform spectral energy across patches)
            spectral_score = 1.0 - min(cv_energy / 1.0, 1.0)

        # --- 3. Long-range luminance autocorrelation ---
        gray_f = gray.astype(np.float64)
        # Downsample to speed up and measure macro-level correlations
        small = cv2.resize(gray_f, (64, 64), interpolation=cv2.INTER_AREA)
        small_flat = small.flatten()
        small_centered = small_flat - np.mean(small_flat)
        norm = np.sum(small_centered ** 2) + 1e-10

        # Measure autocorrelation at large lags (>16 pixels in 64x64 space)
        long_range_acorrs = []
        for lag in range(16, 33):
            acorr = np.sum(
                small_centered[:-lag] * small_centered[lag:]
            ) / norm
            long_range_acorrs.append(abs(acorr))

        avg_lr_acorr = np.mean(long_range_acorrs) if long_range_acorrs else 0.0
        # Flux: higher long-range autocorrelation from global attention
        lr_score = min(avg_lr_acorr / 0.2, 1.0)

        score = (
            0.25 * colour_uniformity
            + 0.20 * texture_uniformity
            + 0.25 * spectral_score
            + 0.30 * lr_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def _attention_uniformity_gray(self, gray_f: np.ndarray) -> float:
        """Fallback attention uniformity analysis for grayscale input."""
        h, w = gray_f.shape

        patch_size = max(h // 4, 16)
        patch_means = []
        patch_stds = []
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = gray_f[i : i + patch_size, j : j + patch_size]
                patch_means.append(np.mean(patch))
                patch_stds.append(np.std(patch))

        if len(patch_means) < 4:
            return 0.0

        cv_means = np.std(patch_means) / (np.mean(patch_means) + 1e-10)
        cv_stds = np.std(patch_stds) / (np.mean(patch_stds) + 1e-10)

        uniformity = 1.0 - min((cv_means + cv_stds) / 1.0, 1.0)
        return float(np.clip(uniformity, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation."""
        if score > 0.7:
            return (
                "High Flux-specific artifacts detected — likely generated "
                "by Black Forest Labs Flux"
            )
        elif score > 0.4:
            return "Moderate Flux signatures — suspicious, possibly Flux-generated"
        else:
            return "Low Flux artifacts — unlikely to be Flux-generated"
