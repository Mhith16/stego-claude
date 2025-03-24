import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia.filters as K  # If you have kornia installed

class NoiseLayer(nn.Module):
    def __init__(self, noise_types=None):
        super(NoiseLayer, self).__init__()
        self.noise_types = noise_types or ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']
        
        # Default noise parameters
        self.dropout_prob = 0.5
        self.jpeg_quality = 50
        self.gaussian_std = 0.05
        self.blur_sigma = 1.0
        self.salt_pepper_density = 0.1
        
    def update_parameters(self, dropout_prob=None, jpeg_quality=None, gaussian_std=None, 
                         blur_sigma=None, salt_pepper_density=None):
        """Update the parameters for different noise types"""
        if dropout_prob is not None:
            self.dropout_prob = float(dropout_prob)
        if jpeg_quality is not None:
            self.jpeg_quality = float(jpeg_quality)
        if gaussian_std is not None:
            self.gaussian_std = float(gaussian_std)
        if blur_sigma is not None:
            self.blur_sigma = float(blur_sigma)
        if salt_pepper_density is not None:
            self.salt_pepper_density = float(salt_pepper_density)
        
        print(f"Noise parameters updated: dropout={self.dropout_prob}, jpeg={self.jpeg_quality}, "
              f"gaussian={self.gaussian_std}, blur={self.blur_sigma}, salt_pepper={self.salt_pepper_density}")
        
    def forward(self, x, noise_type=None):
        # If no specific noise type is specified, randomly choose one
        if noise_type is None:
            noise_type = np.random.choice(self.noise_types)
        
        # Apply the selected noise
        if noise_type == 'dropout':
            return self.dropout(x)
        elif noise_type == 'jpeg':
            return self.jpeg_compression(x)
        elif noise_type == 'gaussian':
            return self.gaussian_noise(x)
        elif noise_type == 'blur':
            return self.gaussian_blur(x)
        elif noise_type == 'salt_pepper':
            return self.salt_pepper_noise(x)
        else:
            return x  # No noise
    
    def dropout(self, x, dropout_prob=None):
        """Random pixel dropout"""
        prob = dropout_prob if dropout_prob is not None else self.dropout_prob
        mask = torch.rand_like(x) > prob
        return x * mask
    
    def jpeg_compression(self, x, quality_factor=None):
        """Simulate JPEG compression"""
        # Use the class parameter if none is provided
        quality = quality_factor if quality_factor is not None else self.jpeg_quality
        
        # This is a simplified simulation
        # For more accurate JPEG simulation, consider using differentiable JPEG libraries
        
        # Convert to YCbCr (approximate)
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 0:1] + 0.114 * x[:, 0:1]
        
        # DCT approximation (using fixed masks for high frequencies)
        batch_size, _, h, w = y.shape
        
        # Create blocks of 8x8
        y_blocks = y.unfold(2, 8, 8).unfold(3, 8, 8)
        y_blocks = y_blocks.contiguous().view(-1, 8, 8)
        
        # Simple simulation of quantization by zeroing out high frequencies
        mask = torch.ones(8, 8, device=y.device)
        threshold = max(1, int(8 * (1 - quality / 100)))
        for i in range(8):
            for j in range(8):
                if i + j >= 8 - threshold:
                    mask[i, j] = 0
        
        # Apply mask (zero out high frequencies)
        y_blocks = y_blocks * mask
        
        # Reshape back
        y_blocks = y_blocks.view(batch_size, 1, h // 8, w // 8, 8, 8)
        y_compressed = y_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, 1, h, w)
        
        # For grayscale, just return the Y channel
        return y_compressed
    
    def gaussian_noise(self, x, std=None):
        """Add Gaussian noise"""
        # Use the class parameter if none is provided
        sigma = std if std is not None else self.gaussian_std
        
        noise = torch.randn_like(x) * sigma
        noisy = x + noise
        return torch.clamp(noisy, 0, 1)
    
    def gaussian_blur(self, x, kernel_size=5, sigma=None):
        """Apply Gaussian blur"""
        # Use the class parameter if none is provided
        blur_sigma = sigma if sigma is not None else self.blur_sigma
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create 1D Gaussian kernel
        channels = x.shape[1]
        
        # Create Gaussian kernel manually
        # First create a 1D kernel
        half_size = kernel_size // 2
        kernel_1d = torch.exp(-torch.arange(-half_size, half_size+1, dtype=torch.float, device=x.device) ** 2 / (2 * blur_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
        
        # Apply padding
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Apply convolution with the Gaussian kernel
        # We need to group the channels since we want to apply the same Gaussian filter to each channel
        blurred = F.conv2d(x_padded, kernel_2d, groups=channels)
        
        return blurred
    
    def salt_pepper_noise(self, x, density=None):
        """Add salt and pepper noise"""
        # Use the class parameter if none is provided
        pepper_salt_density = density if density is not None else self.salt_pepper_density
        
        noise = torch.rand_like(x)
        
        # Salt (white) noise
        salt = (noise < pepper_salt_density/2).float()
        
        # Pepper (black) noise
        pepper = (noise > 1 - pepper_salt_density/2).float()
        
        # Apply salt and pepper noise
        noisy = x.clone()
        noisy[salt > 0] = 1.0
        noisy[pepper > 0] = 0.0
        
        return noisy