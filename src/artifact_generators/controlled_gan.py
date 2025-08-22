import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from typing import List, Tuple

class SimpleFaceGenerator(nn.Module):
    """Basic generator for creating faces with controllable artifacts"""
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()
        self.img_channels = img_channels
        
        # Simple generator architecture
        self.main = nn.Sequential(
            # Input: latent_dim
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 8, 8)),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, img_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

class AdaptiveLossGAN:
    """GAN with adaptive loss weights to generate specific artifacts"""
    def __init__(self, device='cpu'):
        self.device = device
        self.generator = SimpleFaceGenerator().to(device)
        self.latent_dim = 100
        
        # Loss weight configurations for different artifact types
        self.loss_configs = {
            'high_pixel': {'alpha': 0.9, 'beta': 0.05, 'gamma': 0.05},
            'high_perceptual': {'alpha': 0.1, 'beta': 0.8, 'gamma': 0.1},
            'high_adversarial': {'alpha': 0.1, 'beta': 0.1, 'gamma': 0.8}
        }
    
    def generate_with_artifacts(self, artifact_type: str, num_samples: int = 10) -> List[np.ndarray]:
        """Generate samples with specific artifact patterns"""
        self.generator.eval()
        generated_faces = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random latent vector
                z = torch.randn(1, self.latent_dim).to(self.device)
                fake_img = self.generator(z)
                
                # Convert to numpy and post-process based on artifact type
                img = fake_img.squeeze().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = (img + 1) / 2.0  # Denormalize from [-1,1] to [0,1]
                
                # Apply artifact-specific post-processing
                if artifact_type == 'high_pixel':
                    # Simulate over-smoothing from high pixel loss
                    img = cv2.GaussianBlur(img, (3, 3), 0.5)
                elif artifact_type == 'high_perceptual':
                    # Simulate texture artifacts from perceptual loss
                    img = self._add_texture_artifacts(img)
                elif artifact_type == 'high_adversarial':
                    # Simulate mode collapse patterns
                    img = self._add_mode_collapse_artifacts(img)
                
                # Convert to uint8 and resize
                img = (img * 255).astype(np.uint8)
                img = cv2.resize(img, (224, 224))
                generated_faces.append(img)
        
        return generated_faces
    
    def _add_texture_artifacts(self, img):
        """Add texture inconsistencies typical of perceptual loss issues"""
        # Add slight noise in frequency domain for each channel
        img_artifacts = img.copy()
        for c in range(img.shape[2]):
            img_freq = np.fft.fft2(img[:,:,c])
            noise = np.random.normal(0, 0.01, img_freq.shape)
            img_freq_noisy = img_freq + noise
            img_artifacts[:,:,c] = np.real(np.fft.ifft2(img_freq_noisy))
        return np.clip(img_artifacts, 0, 1)
    
    def _add_mode_collapse_artifacts(self, img):
        """Add repetitive patterns typical of mode collapse"""
        # Create subtle repetitive patterns for each channel
        h, w, c = img.shape
        pattern = np.sin(np.linspace(0, 4*np.pi, w)) * 0.02
        
        for channel in range(c):
            for i in range(h):
                img[i, :, channel] += pattern
        
        return np.clip(img, 0, 1)

def create_artifact_samples():
    """Generate samples with known artifacts for training"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    gan = AdaptiveLossGAN(device)
    
    # Create output directory
    output_dir = "../../data/generated"
    os.makedirs(output_dir, exist_ok=True)
    
    artifact_types = ['high_pixel', 'high_perceptual', 'high_adversarial']
    
    for artifact_type in artifact_types:
        print(f"Generating {artifact_type} samples...")
        samples = gan.generate_with_artifacts(artifact_type, num_samples=20)
        
        # Save samples
        type_dir = os.path.join(output_dir, artifact_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            filename = f"{artifact_type}_{i:03d}.jpg"
            cv2.imwrite(os.path.join(type_dir, filename), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        
        print(f"Saved {len(samples)} {artifact_type} samples")
    
    print("Artifact generation complete!")

if __name__ == "__main__":
    create_artifact_samples()
