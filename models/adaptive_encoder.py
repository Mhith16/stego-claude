import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, irfft2

class AdaptiveResidualBlock(nn.Module):
    """Residual block with adaptive embedding strength"""
    def __init__(self, in_channels, out_channels):
        super(AdaptiveResidualBlock, self).__init__()
        
        # Two convolutional layers with batch norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection if dimensions don't match
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Non-inplace addition
        return F.relu(out)

class FrequencyDomainEmbedding(nn.Module):
    """Embeds data in frequency domain coefficients"""
    def __init__(self):
        super(FrequencyDomainEmbedding, self).__init__()
        
    def forward(self, image, message, strength_map):
        # Convert to frequency domain
        freq = rfft2(image)
        
        # Create message mask in proper shape
        B, C, H, W = image.shape
        message_expanded = message.view(B, 1, -1)
        message_reshaped = F.interpolate(message_expanded.unsqueeze(3), 
                                        size=(H, W//2 + 1), 
                                        mode='bilinear', 
                                        align_corners=False)
        message_reshaped = message_reshaped.squeeze(3)
        
        # Apply strength map - stronger in high-texture areas
        strength = strength_map * 0.1  # Scale down embedding strength
        
        # Only modify a portion of mid-frequency coefficients
        # This preserves visual quality better than modifying all frequencies
        start_h, end_h = H//8, H//4
        start_w, end_w = W//8, W//4
        
        # Apply embedding in selected frequency range
        mask = torch.zeros_like(freq)
        mask[:, :, start_h:end_h, start_w:end_w] = message_reshaped[:, :, start_h:end_h, start_w:end_w] * strength[:, :, start_h:end_h, start_w:end_w]
        
        # Add message to frequency coefficients
        modified_freq = freq + mask
        
        # Convert back to spatial domain
        embedded_image = irfft2(modified_freq, s=(H, W))
        
        # Ensure values are in valid range
        return torch.clamp(embedded_image, 0, 1)

class AdaptiveSteganographyEncoder(nn.Module):
    """Enhanced encoder with adaptive embedding strength and frequency domain options"""
    def __init__(self, image_channels=1, embedding_mode='spatial'):
        super(AdaptiveSteganographyEncoder, self).__init__()
        
        self.embedding_mode = embedding_mode  # 'spatial' or 'frequency'
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, padding=1)
        
        # Process the message - using dynamic layers
        self.fixed_message_length = 1024  # The original size the model expects
        self.prep_msg = nn.Sequential(
            nn.Linear(self.fixed_message_length, self.fixed_message_length),
            nn.ReLU(inplace=True)
        )
        
        # Message embedding layer
        self.embed_msg = nn.Linear(self.fixed_message_length, 64*64)  # Reshape to spatial dimension
        
        # Residual blocks with increasing channels
        self.res_block1 = AdaptiveResidualBlock(32 + 1, 64)  # +1 for message channel
        self.res_block2 = AdaptiveResidualBlock(64, 128)
        self.res_block3 = AdaptiveResidualBlock(128, 64)
        self.res_block4 = AdaptiveResidualBlock(64, 32)
        
        # Final convolution
        self.final_conv = nn.Conv2d(32, image_channels, kernel_size=3, padding=1)
        
        # Frequency domain embedding layer (optional)
        self.freq_embedding = FrequencyDomainEmbedding()
        
        # Dynamic adaptation layers
        self.dynamic_msg_adapter = None
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _adjust_message_size(self, message):
        """Dynamically handle message sizes that don't match what the model expects"""
        # Check if we need to adjust
        if message.size(1) == self.fixed_message_length:
            return message
        
        # Create a dynamic adapter if needed
        if self.dynamic_msg_adapter is None or self.dynamic_msg_adapter.in_features != message.size(1):
            self.dynamic_msg_adapter = nn.Linear(message.size(1), self.fixed_message_length).to(message.device)
            # Initialize with identity-like weights for minimal distortion
            nn.init.zeros_(self.dynamic_msg_adapter.weight)
            min_size = min(message.size(1), self.fixed_message_length)
            for i in range(min_size):
                self.dynamic_msg_adapter.weight[i, i] = 1.0
        
        # Apply the adapter
        return self.dynamic_msg_adapter(message)
        
    def forward(self, image, message, feature_weights=None):
        # If no feature weights provided, use uniform weights
        if feature_weights is None:
            feature_weights = torch.ones_like(image)
            
        # Choose embedding method
        if self.embedding_mode == 'frequency':
            return self.forward_frequency(image, message, feature_weights)
        else:
            return self.forward_spatial(image, message, feature_weights)
            
    def forward_spatial(self, image, message, feature_weights):
        # Initial image processing
        x = self.conv1(image)  # [B, 32, H, W]
        
        # Adjust message size if needed
        adjusted_message = self._adjust_message_size(message)
        
        # Process the message
        msg = self.prep_msg(adjusted_message)  # [B, fixed_message_length]
        
        # Embed message into spatial feature map
        msg_spatial = self.embed_msg(msg)  # [B, 64*64]
        msg_spatial = msg_spatial.view(-1, 1, 64, 64)  # [B, 1, 64, 64]
        
        # Resize to match feature map dimensions
        msg_spatial = F.interpolate(msg_spatial, size=(x.size(2), x.size(3)), 
                                    mode='bilinear', align_corners=False)
        
        # Apply feature weights
        msg_spatial = msg_spatial * feature_weights
        
        # Concatenate message with image features
        x = torch.cat([x, msg_spatial], dim=1)  # [B, 33, H, W]
        
        # Process through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Final processing
        residual = self.final_conv(x)
        
        # Add residual connection with adaptive strength
        # Use feature weights to determine embedding strength
        # Higher values in feature_weights mean stronger embedding
        embedding_strength = torch.mean(feature_weights, dim=1, keepdim=True)
        stego_image = image + torch.tanh(residual) * 0.1 * embedding_strength
        
        # Ensure output values are in valid range [0, 1]
        stego_image = torch.clamp(stego_image, 0, 1)
        
        return stego_image
        
    def forward_frequency(self, image, message, feature_weights):
        """Alternative embedding in frequency domain for better quality"""
        # Initial image processing to get residual
        x = self.conv1(image)
        
        # Adjust message size if needed
        adjusted_message = self._adjust_message_size(message)
        
        # Process the message
        msg = self.prep_msg(adjusted_message)
        
        # Process through residual blocks - this determines HOW to embed
        # rather than the actual embedding
        msg_spatial = self.embed_msg(msg).view(-1, 1, 64, 64)
        msg_spatial = F.interpolate(msg_spatial, size=(x.size(2), x.size(3)), 
                                   mode='bilinear', align_corners=False)
        
        x = torch.cat([x, msg_spatial], dim=1)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Instead of adding residual in spatial domain,
        # embed in frequency domain which can preserve more image details
        stego_image = self.freq_embedding(image, message, feature_weights)
        
        return stego_image