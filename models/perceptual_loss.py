import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 pre-trained on ImageNet
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        blocks = []
        # Use first four blocks of VGG for perceptual loss
        blocks.append(vgg.features[:4].eval())   # relu1_2
        blocks.append(vgg.features[4:9].eval())  # relu2_2
        blocks.append(vgg.features[9:16].eval()) # relu3_3
        blocks.append(vgg.features[16:23].eval()) # relu4_3
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        
        self.blocks = nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, input, target):
        if input.shape[1] == 1:
            # If grayscale, repeat to make 3 channels
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        # Prepare input by normalizing for VGG
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            # VGG expects minimum 224x224 images
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        
        # Calculate loss at multiple layers
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
            
        return loss

class SSIMLoss(nn.Module):
    """
    Structural similarity loss (SSIM)
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        
    def _gaussian(self, window_size, sigma):
        """Create a 1D Gaussian kernel"""
        import math
        # Create Gaussian weights
        gauss = torch.tensor([
            math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ], dtype=torch.float)
        # Normalize
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        """Create a 2D Gaussian kernel"""
        # Create 1D Gaussian window
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        # Create 2D Gaussian window by multiplying 1D window with its transpose
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        # Expand to match the number of channels
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def forward(self, img1, img2):
        # Check image dimensions
        _, channels, height, width = img1.size()
        
        # If window is not the right size, recreate it
        if (self.window.data.type() != img1.data.type() or
            self.window.size(2) != self.window_size or  # Changed from window_size to self.window_size
            self.window.size(1) != channels):
            window = self._create_window(self.window_size, channels)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            
        window = self.window
        
        # Calculate means
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channels) - mu1_mu2
        
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return 1 - SSIM to convert to a loss (0 is best)
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class EnhancedSteganographyLoss(nn.Module):
    """
    Enhanced loss function combining MSE, perceptual, SSIM, and binary cross-entropy
    """
    def __init__(self, lambda_image=1.0, lambda_message=20.0, lambda_adv=0.1, lambda_perceptual=5.0, lambda_ssim=5.0):
        super(EnhancedSteganographyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.ssim_loss = SSIMLoss()
        
        # Loss weights
        self.lambda_image = lambda_image
        self.lambda_message = lambda_message
        self.lambda_adv = lambda_adv
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        
    def encoder_loss(self, original_image, stego_image, disc_output):
        """Calculate encoder loss combining image quality and adversarial components"""
        # Image quality losses
        mse = self.mse_loss(stego_image, original_image)
        perceptual = self.perceptual_loss(stego_image, original_image)
        ssim = self.ssim_loss(stego_image, original_image)
        
        # Adversarial loss (fool the discriminator)
        adv_loss = self.bce_loss(disc_output, torch.ones_like(disc_output))
        
        # Combined loss
        total_loss = (
            self.lambda_image * mse +
            self.lambda_perceptual * perceptual +
            self.lambda_ssim * ssim +
            self.lambda_adv * adv_loss
        )
        
        # Return total and components for logging
        return {
            'total': total_loss,
            'mse': mse,
            'perceptual': perceptual,
            'ssim': ssim,
            'adversarial': adv_loss
        }
        
    def decoder_loss(self, original_message, decoded_message, confidence=None):
        """Calculate decoder loss with optional field-specific weighting"""
        # Handle size mismatch - either truncate or pad
        if original_message.size(1) != decoded_message.size(1):
            print(f"Warning: Size mismatch - original: {original_message.size()}, decoded: {decoded_message.size()}")
            if original_message.size(1) < decoded_message.size(1):
                # Pad original message with zeros to match decoded size
                padding = torch.zeros(original_message.size(0), 
                                    decoded_message.size(1) - original_message.size(1), 
                                    device=original_message.device)
                original_message = torch.cat([original_message, padding], dim=1)
            else:
                # Truncate decoded message to match original size
                decoded_message = decoded_message[:, :original_message.size(1)]
        
        # If confidence values are provided and we have field-specific extraction
        if confidence is not None and isinstance(decoded_message, dict):
            name_loss = self.bce_loss(decoded_message['fields']['name'], 
                                    original_message[:, :decoded_message['fields']['name'].size(1)])
            
            age_loss = self.bce_loss(decoded_message['fields']['age'], 
                                    original_message[:, decoded_message['fields']['name'].size(1):
                                                    decoded_message['fields']['name'].size(1) + 
                                                    decoded_message['fields']['age'].size(1)])
            
            id_loss = self.bce_loss(decoded_message['fields']['id'], 
                                original_message[:, decoded_message['fields']['name'].size(1) + 
                                                    decoded_message['fields']['age'].size(1):])
            
            # Weight losses by importance - name and ID are more critical than age
            message_loss = name_loss * 1.2 + age_loss * 0.8 + id_loss * 1.0
        else:
            # Standard loss for entire message
            message_loss = self.bce_loss(decoded_message, original_message)
        
        # Apply global message weight
        total_loss = self.lambda_message * message_loss
        
        return {
            'total': total_loss,
            'message': message_loss
        }
        
    def discriminator_loss(self, real_output, fake_output):
        """Standard discriminator loss"""
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        total_loss = (real_loss + fake_loss) / 2
        
        return {
            'total': total_loss,
            'real': real_loss,
            'fake': fake_loss
        }