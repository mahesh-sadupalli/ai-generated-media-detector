import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple

class SmoothingArtifactDetector:
    """Detects over-smoothing artifacts from high pixel-loss GAN training"""
    
    def __init__(self):
        self.threshold = 0.5
    
    def detect_smoothing_artifacts(self, image: np.ndarray) -> Tuple[float, dict]:
        """
        Detect smoothing artifacts in face image
        
        Args:
            image: Face image (224x224x3)
            
        Returns:
            score: Smoothing artifact score (0-1, higher = more artifacts)
            details: Dictionary with detailed analysis
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate multiple smoothness metrics
        texture_variance = self._calculate_texture_variance(gray)
        edge_sharpness = self._calculate_edge_sharpness(gray)
        local_std = self._calculate_local_std(gray)
        frequency_content = self._calculate_frequency_content(gray)
        
        # Combine metrics (lower values indicate more smoothing)
        smoothing_score = 1.0 - (
            0.3 * texture_variance + 
            0.3 * edge_sharpness + 
            0.2 * local_std + 
            0.2 * frequency_content
        )
        
        smoothing_score = np.clip(smoothing_score, 0, 1)
        
        details = {
            'texture_variance': texture_variance,
            'edge_sharpness': edge_sharpness,
            'local_std': local_std,
            'frequency_content': frequency_content,
            'interpretation': self._interpret_score(smoothing_score)
        }
        
        return smoothing_score, details
    
    def _calculate_texture_variance(self, gray_image: np.ndarray) -> float:
        """Calculate local texture variance"""
        # Use local binary patterns for texture analysis
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        variance = np.mean((gray_image.astype(np.float32) - local_mean) ** 2)
        
        # Normalize to 0-1 range
        return min(variance / 1000.0, 1.0)
    
    def _calculate_edge_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate edge sharpness using gradient magnitude"""
        # Sobel edge detection
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Mean gradient magnitude
        sharpness = np.mean(gradient_magnitude)
        
        # Normalize to 0-1 range
        return min(sharpness / 50.0, 1.0)
    
    def _calculate_local_std(self, gray_image: np.ndarray) -> float:
        """Calculate local standard deviation"""
        # Use sliding window to calculate local std
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray_image.astype(np.float32))**2, -1, kernel)
        local_var = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Mean local standard deviation
        mean_local_std = np.mean(local_std)
        
        # Normalize to 0-1 range
        return min(mean_local_std / 30.0, 1.0)
    
    def _calculate_frequency_content(self, gray_image: np.ndarray) -> float:
        """Analyze frequency domain content"""
        # FFT to frequency domain
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Calculate high frequency content
        h, w = magnitude_spectrum.shape
        center_h, center_w = h//2, w//2
        
        # Create high frequency mask (outer region)
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h,w)//4)**2
        
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Normalize to 0-1 range
        return min(high_freq_ratio * 10, 1.0)
    
    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation"""
        if score > 0.7:
            return "High smoothing artifacts detected - likely generated content"
        elif score > 0.4:
            return "Moderate smoothing artifacts - suspicious"
        else:
            return "Low smoothing artifacts - appears natural"

def test_smoothing_detector():
    """Test smoothing detector on real vs generated samples"""
    detector = SmoothingArtifactDetector()
    
    # Test on real face from FaceForensics++
    real_face_path = "../../data/raw/faceforensics/original_sequences/youtube/c23/videos/183.mp4"
    
    if os.path.exists(real_face_path):
        from ..utils.simple_face_detection import SimpleFaceDetector
        face_extractor = SimpleFaceDetector()
        real_faces = face_extractor.extract_faces_from_video(real_face_path, max_frames=1)
        
        if real_faces:
            score, details = detector.detect_smoothing_artifacts(real_faces[0])
            print(f"Real face smoothing score: {score:.3f}")
            print(f"Interpretation: {details['interpretation']}")
            print(f"Details: {details}")

if __name__ == "__main__":
    import os
    test_smoothing_detector()
