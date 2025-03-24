import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SelfAttention(nn.Module):
    """Self attention layer for feature refinement"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Reshape for attention calculation
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Residual connection with learned weight
        out = self.gamma * out + x
        return out

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales"""
    def __init__(self, in_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Different kernel sizes for multi-scale extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 conv to aggregate features
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate features from different scales
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
        return self.conv_final(outputs)

class EnhancedFeatureAnalysisDenseNet(nn.Module):
    """Enhanced feature analyzer using pretrained DenseNet with attention and multi-scale features"""
    def __init__(self, in_channels=1, pretrained=True):
        super(EnhancedFeatureAnalysisDenseNet, self).__init__()
        
        # Load pretrained DenseNet121 (trained on medical images)
        densenet = models.densenet121(pretrained=pretrained)
        
        # Adapt first layer to grayscale input if needed
        if in_channels == 1:
            self.first_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize with the average of RGB channels
            with torch.no_grad():
                if pretrained:
                    weight = densenet.features.conv0.weight.sum(dim=1, keepdim=True)
                    self.first_conv.weight.copy_(weight)
        else:
            self.first_conv = densenet.features.conv0
        
        # Extract the feature layers we need from DenseNet
        self.backbone = nn.Sequential(
            self.first_conv,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
            densenet.features.denseblock1,
            densenet.features.transition1,
            densenet.features.denseblock2
        )
        
        # Get the number of features after the backbone
        feature_channels = 512  # Approximate number after denseblock2
        
        # Add attention mechanism
        self.attention = SelfAttention(feature_channels)
        
        # Add multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(feature_channels)
        
        # Upsampling path to restore original resolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution to get a single-channel feature map
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        # Sigmoid to get values in [0,1] range representing embedding capacity
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input size for reference
        input_size = x.size()[2:]
        
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Apply attention to focus on important regions
        attended_features = self.attention(features)
        
        # Apply multi-scale feature extraction
        multi_scale_features = self.multi_scale(attended_features)
        
        # Upsampling path to restore original resolution
        x = self.up1(multi_scale_features)
        x = self.up2(x)
        x = self.up3(x)
        
        # Ensure the output size matches the input size
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # Final convolution and sigmoid activation
        capacity_map = self.final_conv(x)
        capacity_map = self.sigmoid(capacity_map)
        
        return capacity_map

    def load_pretrained_weights(self, weights_path):
        """Load pretrained weights from ChestX-ray datasets if available"""
        try:
            state_dict = torch.load(weights_path)
            # Filter out any keys that don't match our model
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict()}
            self.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Using default ImageNet weights instead")