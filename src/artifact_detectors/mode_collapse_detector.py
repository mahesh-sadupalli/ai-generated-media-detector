import cv2
import numpy as np
from scipy import stats
from typing import Tuple

class ModeCollapseDetector:
    """Detects mode collapse artifacts from high adversarial-loss GAN training"""
    
    def __init__(self):
        self.threshold = 0.5
    
    def detect_mode_collapse_artifacts(self, image: np.ndarray) -> Tuple[float, dict]:
        """
        Detect mode collapse patterns in face image
        
        Args:
            image: Face image (224x224x3)
            
        Returns:
            score: Mode collapse artifact score (0-1, higher = more artifacts)
            details: Dictionary with detailed analysis
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate mode collapse indicators
        symmetry_score = self._analyze_unnatural_symmetry(gray)
        repetition_score = self._analyze_feature_repetition(gray)
        diversity_score = self._analyze_feature_diversity(image)
        instability_score = self._analyze_training_instability_patterns(gray)
        
        # Combine metrics
        collapse_score = (
            0.3 * symmetry_score + 
            0.3 * repetition_score + 
            0.2 * diversity_score + 
            0.2 * instability_score
        )
        
        collapse_score = np.clip(collapse_score, 0, 1)
        
        details = {
            'symmetry_score': symmetry_score,
            'repetition_score': repetition_score,
            'diversity_score': diversity_score,
            'instability_score': instability_score,
            'interpretation': self._interpret_score(collapse_score)
        }
        
        return collapse_score, details
    
    def _analyze_unnatural_symmetry(self, gray_image: np.ndarray) -> float:
        """Detect unnatural facial symmetry patterns"""
        h, w = gray_image.shape
        
        # Split image into left and right halves
        left_half = gray_image[:, :w//2]
        right_half = gray_image[:, w//2:]
        right_half_flipped = np.fliplr(right_half)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Calculate correlation between halves
        correlation = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
        correlation = np.nan_to_num(correlation)
        
        # Perfect symmetry is unnatural for real faces
        # Mode collapse often creates overly symmetric features
        symmetry_unnaturalness = abs(correlation) if abs(correlation) > 0.7 else 0
        
        return symmetry_unnaturalness
    
    def _analyze_feature_repetition(self, gray_image: np.ndarray) -> float:
        """Detect repetitive patterns typical of mode collapse"""
        # Analyze horizontal and vertical patterns
        h, w = gray_image.shape
        
        repetition_scores = []
        
        # Check for horizontal repetitions
        for row in range(0, h, 10):  # Sample every 10th row
            if row < h:
                row_data = gray_image[row, :]
                # Calculate autocorrelation
                autocorr = np.correlate(row_data, row_data, mode='full')
                autocorr = autocorr[autocorr.size//2:]
                
                # Look for peaks indicating repetition
                if len(autocorr) > 10:
                    normalized_autocorr = autocorr[1:] / (autocorr[0] + 1e-10)
                    max_correlation = np.max(normalized_autocorr[:len(normalized_autocorr)//2])
                    repetition_scores.append(max_correlation)
        
        # Check for vertical repetitions
        for col in range(0, w, 10):  # Sample every 10th column
            if col < w:
                col_data = gray_image[:, col]
                autocorr = np.correlate(col_data, col_data, mode='full')
                autocorr = autocorr[autocorr.size//2:]
                
                if len(autocorr) > 10:
                    normalized_autocorr = autocorr[1:] / (autocorr[0] + 1e-10)
                    max_correlation = np.max(normalized_autocorr[:len(normalized_autocorr)//2])
                    repetition_scores.append(max_correlation)
        
        # Average repetition score
        avg_repetition = np.mean(repetition_scores) if repetition_scores else 0
        return min(avg_repetition, 1.0)
    
    def _analyze_feature_diversity(self, image: np.ndarray) -> float:
        """Analyze diversity of facial features"""
        if len(image.shape) != 3:
            return 0.0
        
        # Convert to different color spaces and analyze diversity
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        diversity_scores = []
        
        # Analyze each channel
        for channel in range(3):
            channel_data = hsv[:, :, channel].flatten()
            
            # Calculate histogram
            hist, _ = np.histogram(channel_data, bins=32, range=(0, 256))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-10)
            
            # Calculate entropy (diversity measure)
            entropy = stats.entropy(hist + 1e-10)
            diversity_scores.append(entropy)
        
        # Low diversity indicates mode collapse
        avg_diversity = np.mean(diversity_scores)
        max_entropy = np.log(32)  # Maximum possible entropy for 32 bins
        
        # Invert so high score = low diversity = more mode collapse
        diversity_collapse_score = 1.0 - (avg_diversity / max_entropy)
        
        return diversity_collapse_score
    
    def _analyze_training_instability_patterns(self, gray_image: np.ndarray) -> float:
        """Detect patterns that suggest training instability"""
        # Calculate gradient patterns that might indicate instability
        
        # First and second order gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Second order gradients
        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and curvature
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        curvature = np.abs(grad_xx + grad_yy)
        
        # Analyze gradient statistics
        grad_variance = np.var(grad_mag)
        curvature_variance = np.var(curvature)
        
        # Training instability often creates irregular gradient patterns
        instability_indicator = min((grad_variance + curvature_variance) / 10000.0, 1.0)
        
        return instability_indicator
    
    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation"""
        if score > 0.7:
            return "High mode collapse artifacts - repetitive features detected"
        elif score > 0.4:
            return "Moderate mode collapse indicators"
        else:
            return "Natural feature diversity observed"

def test_mode_collapse_detector():
    """Test mode collapse detector"""
    detector = ModeCollapseDetector()
    
    # Test on a sample image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    score, details = detector.detect_mode_collapse_artifacts(test_img)
    
    print(f"Test mode collapse score: {score:.3f}")
    print(f"Details: {details}")

if __name__ == "__main__":
    test_mode_collapse_detector()
