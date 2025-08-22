import cv2
import numpy as np
from scipy import stats
from typing import Tuple

class TextureArtifactDetector:
    """Detects texture inconsistencies from high perceptual-loss GAN training"""
    
    def __init__(self):
        self.threshold = 0.5
    
    def detect_texture_artifacts(self, image: np.ndarray) -> Tuple[float, dict]:
        """
        Detect texture artifacts in face image
        
        Args:
            image: Face image (224x224x3)
            
        Returns:
            score: Texture artifact score (0-1, higher = more artifacts)
            details: Dictionary with detailed analysis
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate texture analysis metrics
        frequency_anomalies = self._analyze_frequency_domain(gray)
        lbp_inconsistency = self._analyze_local_binary_patterns(gray)
        color_distribution = self._analyze_color_distribution(image)
        texture_regularity = self._analyze_texture_regularity(gray)
        
        # Combine metrics
        texture_score = (
            0.3 * frequency_anomalies + 
            0.3 * lbp_inconsistency + 
            0.2 * color_distribution + 
            0.2 * texture_regularity
        )
        
        texture_score = np.clip(texture_score, 0, 1)
        
        details = {
            'frequency_anomalies': frequency_anomalies,
            'lbp_inconsistency': lbp_inconsistency,
            'color_distribution': color_distribution,
            'texture_regularity': texture_regularity,
            'interpretation': self._interpret_score(texture_score)
        }
        
        return texture_score, details
    
    def _analyze_frequency_domain(self, gray_image: np.ndarray) -> float:
        """Analyze frequency domain for artifacts from perceptual loss"""
        # FFT analysis
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate frequency distribution statistics
        freq_mean = np.mean(magnitude_spectrum)
        freq_std = np.std(magnitude_spectrum)
        
        # Analyze spectral anomalies
        # High perceptual loss can create unnatural frequency patterns
        spectral_entropy = stats.entropy(magnitude_spectrum.flatten() + 1e-10)
        
        # Normalize and combine
        freq_anomaly = 1.0 - min(spectral_entropy / 15.0, 1.0)
        
        return freq_anomaly
    
    def _analyze_local_binary_patterns(self, gray_image: np.ndarray) -> float:
        """Analyze texture using Local Binary Patterns"""
        def lbp(img, radius=1, neighbors=8):
            """Simple LBP implementation"""
            h, w = img.shape
            lbp_image = np.zeros_like(img)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = img[i, j]
                    binary_string = ''
                    
                    # Sample neighbors
                    for n in range(neighbors):
                        angle = 2 * np.pi * n / neighbors
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if 0 <= x < h and 0 <= y < w:
                            binary_string += '1' if img[x, y] >= center else '0'
                    
                    lbp_image[i, j] = int(binary_string, 2) if binary_string else 0
            
            return lbp_image
        
        lbp_img = lbp(gray_image)
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp_img, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)
        
        # Calculate uniformity - natural textures have more uniform LBP distributions
        uniformity = 1.0 - stats.entropy(hist + 1e-10) / np.log(256)
        
        return uniformity
    
    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution anomalies"""
        if len(image.shape) != 3:
            return 0.0
        
        # Convert to LAB color space for perceptual analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        color_anomalies = 0.0
        
        # Analyze each channel
        for channel in range(3):
            channel_data = lab[:, :, channel].flatten()
            
            # Calculate distribution statistics
            skewness = abs(stats.skew(channel_data))
            kurtosis = abs(stats.kurtosis(channel_data))
            
            # High perceptual loss can create unnatural color distributions
            channel_anomaly = min((skewness + kurtosis) / 10.0, 1.0)
            color_anomalies += channel_anomaly
        
        return color_anomalies / 3.0
    
    def _analyze_texture_regularity(self, gray_image: np.ndarray) -> float:
        """Analyze texture regularity patterns"""
        # Calculate gray-level co-occurrence matrix properties
        def glcm_contrast(img, d=1):
            """Simplified GLCM contrast calculation"""
            # Quantize image
            img_q = (img / 32).astype(int)
            levels = 8
            
            # Initialize co-occurrence matrix
            glcm = np.zeros((levels, levels))
            
            # Calculate horizontal co-occurrence
            for i in range(img_q.shape[0]):
                for j in range(img_q.shape[1] - d):
                    if img_q[i, j] < levels and img_q[i, j + d] < levels:
                        glcm[img_q[i, j], img_q[i, j + d]] += 1
            
            # Normalize
            glcm = glcm / (glcm.sum() + 1e-10)
            
            # Calculate contrast
            contrast = 0
            for i in range(levels):
                for j in range(levels):
                    contrast += glcm[i, j] * (i - j) ** 2
            
            return contrast
        
        contrast = glcm_contrast(gray_image)
        
        # Normalize contrast - unnatural textures often have extreme contrast values
        regularity_score = min(contrast / 50.0, 1.0)
        
        return regularity_score
    
    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation"""
        if score > 0.7:
            return "High texture artifacts detected - perceptual loss issues"
        elif score > 0.4:
            return "Moderate texture inconsistencies detected"
        else:
            return "Natural texture patterns observed"

def test_texture_detector():
    """Test texture detector"""
    detector = TextureArtifactDetector()
    
    # Test on a sample image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    score, details = detector.detect_texture_artifacts(test_img)
    
    print(f"Test texture score: {score:.3f}")
    print(f"Details: {details}")

if __name__ == "__main__":
    test_texture_detector()
