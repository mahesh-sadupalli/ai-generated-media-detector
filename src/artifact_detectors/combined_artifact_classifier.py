import cv2
import numpy as np
from typing import Dict
from pathlib import Path

from .smoothing_detector import SmoothingArtifactDetector
from .texture_detector import TextureArtifactDetector
from .mode_collapse_detector import ModeCollapseDetector
from .diffusion_detector import DiffusionArtifactDetector
from ..utils.simple_face_detection import SimpleFaceDetector
from ..utils.compression_estimator import CompressionEstimator


class ArtifactClassifier:
    """Combined classifier using GAN and diffusion artifact detectors.

    Produces a 3-class prediction: REAL / GAN-GENERATED / DIFFUSION-GENERATED,
    using a weighted ensemble of four hand-crafted detectors.

    Parameters
    ----------
    thresholds : dict, optional
        Override default classification thresholds.
    """

    def __init__(self, thresholds: Dict[str, float] | None = None) -> None:
        # GAN detectors
        self.smoothing_detector = SmoothingArtifactDetector()
        self.texture_detector = TextureArtifactDetector()
        self.mode_collapse_detector = ModeCollapseDetector()

        # Diffusion detector
        self.diffusion_detector = DiffusionArtifactDetector()

        # Compression estimator (utility, not a detector)
        self.compression_estimator = CompressionEstimator()

        self.thresholds = {
            'smoothing': 0.65,
            'mode_collapse': 0.70,
            'diffusion': 0.55,
            'fake_confidence': 0.60,
        }
        if thresholds:
            self.thresholds.update(thresholds)

    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyse a single face image for AI-generation artifacts.

        Parameters
        ----------
        image : np.ndarray
            Face image (H, W, 3) in RGB colour space.

        Returns
        -------
        dict
            Prediction, confidence, per-detector scores, details,
            and human-readable explanation.
        """
        # Run all detectors
        smoothing_score, smoothing_details = (
            self.smoothing_detector.detect_smoothing_artifacts(image)
        )
        texture_score, texture_details = (
            self.texture_detector.detect_texture_artifacts(image)
        )
        collapse_score, collapse_details = (
            self.mode_collapse_detector.detect_mode_collapse_artifacts(image)
        )
        diffusion_score, diffusion_details = (
            self.diffusion_detector.detect_diffusion_artifacts(image)
        )

        # Estimate compression level and attenuation factor
        compression_level, compression_details = (
            self.compression_estimator.estimate_compression(image)
        )
        attenuation = self.compression_estimator.get_attenuation_factor(
            compression_level
        )

        # Attenuate compression-sensitive sub-scores
        smoothing_adj = smoothing_score * attenuation
        recon_adj = diffusion_details['reconstruction_error'] * attenuation
        noise_adj = diffusion_details['noise_residual'] * attenuation

        # Recompute diffusion score with attenuated sub-scores
        diffusion_adj = (
            0.30 * recon_adj
            + 0.30 * diffusion_details['spectral_fingerprint']
            + 0.20 * noise_adj
            + 0.20 * diffusion_details['patch_consistency']
        )
        diffusion_adj = float(np.clip(diffusion_adj, 0.0, 1.0))

        # GAN score with attenuated smoothing
        gan_score = (
            0.5 * smoothing_adj
            + 0.1 * texture_score
            + 0.4 * collapse_score
        )

        # 3-class decision uses adjusted scores
        prediction, artifact_type = self._classify(
            gan_score, diffusion_adj, smoothing_adj, collapse_score
        )

        # Overall confidence is the max of the two generation scores
        confidence = max(gan_score, diffusion_adj)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'artifact_type': artifact_type,
            'scores': {
                'smoothing': smoothing_score,
                'smoothing_adjusted': smoothing_adj,
                'texture': texture_score,
                'mode_collapse': collapse_score,
                'diffusion': diffusion_score,
                'diffusion_adjusted': diffusion_adj,
                'gan_overall': gan_score,
                'compression_level': compression_level,
                'compression_attenuation': attenuation,
            },
            'details': {
                'smoothing': smoothing_details,
                'texture': texture_details,
                'mode_collapse': collapse_details,
                'diffusion': diffusion_details,
                'compression': compression_details,
            },
            'explanation': self._generate_explanation(
                prediction, artifact_type,
                smoothing_adj, texture_score, collapse_score, diffusion_adj,
            ),
        }

    def analyze_video(self, video_path: str, max_frames: int = 10) -> Dict:
        """Analyse a video for deepfake artifacts.

        Parameters
        ----------
        video_path : str
            Path to video file.
        max_frames : int
            Maximum number of frames to sample.

        Returns
        -------
        dict
            Video-level aggregated analysis.
        """
        face_extractor = SimpleFaceDetector()
        faces = face_extractor.extract_faces_from_video(
            video_path, max_frames=max_frames
        )

        if not faces:
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'explanation': 'No faces detected in video',
            }

        frame_results = [self.analyze_image(face) for face in faces]

        avg_confidence = float(
            np.mean([r['confidence'] for r in frame_results])
        )

        # Count votes per class
        votes = {'REAL': 0, 'GAN-GENERATED': 0, 'DIFFUSION-GENERATED': 0}
        for r in frame_results:
            pred = r['prediction']
            votes[pred] = votes.get(pred, 0) + 1

        video_prediction = max(votes, key=votes.get)

        artifact_types = [
            r['artifact_type'] for r in frame_results
            if r['artifact_type'] != 'unknown'
        ]
        most_common_artifact = (
            max(set(artifact_types), key=artifact_types.count)
            if artifact_types else 'unknown'
        )

        fake_frames = votes.get('GAN-GENERATED', 0) + votes.get('DIFFUSION-GENERATED', 0)

        return {
            'prediction': video_prediction,
            'confidence': avg_confidence,
            'artifact_type': most_common_artifact,
            'frames_analyzed': len(faces),
            'fake_frames': fake_frames,
            'votes': votes,
            'frame_results': frame_results,
            'explanation': (
                f"Video analysis: {fake_frames}/{len(faces)} frames flagged as generated. "
                f"Verdict: {video_prediction}. "
                f"Primary artifact type: {most_common_artifact}"
            ),
        }

    def _classify(
        self,
        gan_score: float,
        diffusion_score: float,
        smoothing_score: float,
        collapse_score: float,
    ) -> tuple:
        """Determine 3-class prediction and artifact type."""
        threshold = self.thresholds['fake_confidence']
        diff_threshold = self.thresholds['diffusion']

        # If both scores are below threshold → REAL
        if gan_score < threshold and diffusion_score < diff_threshold:
            return 'REAL', 'none'

        # If diffusion score dominates → DIFFUSION-GENERATED
        if diffusion_score >= diff_threshold and diffusion_score >= gan_score:
            return 'DIFFUSION-GENERATED', 'diffusion_artifacts'

        # Otherwise → GAN-GENERATED, identify specific artifact type
        artifact_type = 'unknown'
        if smoothing_score > self.thresholds['smoothing']:
            artifact_type = 'pixel_loss_artifacts'
        elif collapse_score > self.thresholds['mode_collapse']:
            artifact_type = 'adversarial_loss_artifacts'
        elif smoothing_score > 0.5 and collapse_score > 0.5:
            artifact_type = 'mixed_artifacts'

        return 'GAN-GENERATED', artifact_type

    def _generate_explanation(
        self,
        prediction: str,
        artifact_type: str,
        smoothing: float,
        texture: float,
        collapse: float,
        diffusion: float,
    ) -> str:
        """Generate a human-readable explanation."""
        gan_overall = 0.5 * smoothing + 0.1 * texture + 0.4 * collapse
        confidence = max(gan_overall, diffusion)

        lines = [f"Prediction: {prediction} (confidence: {confidence:.3f})"]

        if prediction == 'REAL':
            lines.append("Image appears authentic based on artifact analysis.")
        elif prediction == 'DIFFUSION-GENERATED':
            lines.append("Detected diffusion model signatures:")
            lines.append(f"  - Diffusion artifact score: {diffusion:.3f}")
            if diffusion > 0.6:
                lines.append("  - Reconstruction patterns consistent with denoising process")
            if diffusion > 0.5:
                lines.append("  - Spectral fingerprint suggests diffusion model origin")
        else:  # GAN-GENERATED
            lines.append("Detected GAN generation artifacts:")
            if smoothing > 0.65:
                lines.append("  - High smoothing artifacts suggest pixel-loss optimization")
            if collapse > 0.70:
                lines.append("  - Mode collapse patterns suggest adversarial training issues")
            if texture > 0.50:
                lines.append("  - Texture inconsistencies detected")

        lines.append(f"Artifact type: {artifact_type}")
        return '\n'.join(lines)


# Backward compatibility alias
GANArtifactClassifier = ArtifactClassifier
