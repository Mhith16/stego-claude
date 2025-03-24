import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # Normalization and activation before convolution (pre-activation)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return torch.cat([x, out], 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.pool(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        
        self.out_channels = current_channels
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AttentionLayer(nn.Module):
    """Channel-wise attention mechanism"""
    def __init__(self, in_channels, reduction=16):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RedundantSegmentExtractor(nn.Module):
    """Special module to extract redundant data segments"""
    def __init__(self, in_features, segment_size=127):
        super(RedundantSegmentExtractor, self).__init__()
        self.segment_size = segment_size
        
        # MLP to predict segment boundaries
        self.boundary_predictor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)  # Predict up to 8 segment boundaries
        )
        
        # MLP to predict segment quality scores
        self.quality_predictor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 8),  # Quality score for each segment
            nn.Sigmoid()
        )
    
    def forward(self, x, message):
        # Predict segment boundaries
        boundaries = self.boundary_predictor(x)
        boundaries = torch.sigmoid(boundaries) * message.size(1)
        boundaries = boundaries.long()
        
        # Predict quality scores
        quality_scores = self.quality_predictor(x)
        
        return boundaries, quality_scores

class RobustSteganographyDecoder(nn.Module):
    """Enhanced decoder with redundant data handling and confidence estimation"""
    def __init__(self, image_channels=1, message_length=1024, growth_rate=32, block_config=(6, 12, 24, 16), segment_size=127):
        super(RobustSteganographyDecoder, self).__init__()
        self.message_length = message_length
        self.segment_size = segment_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # First channel attention
        self.attn1 = AttentionLayer(64)
        
        # Dense blocks with transitions
        num_features = 64
        self.dense_blocks = nn.ModuleList()
        self.trans_blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.dense_blocks.append(block)
            num_features = block.out_channels
            
            # Add attention after each block
            self.attentions.append(AttentionLayer(num_features))
            
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.trans_blocks.append(trans)
                num_features = num_features // 2
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Main message decoder
        self.message_decoder = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, message_length),
            nn.Sigmoid()
        )
        
        # Add redundant segment extractor
        self.segment_extractor = RedundantSegmentExtractor(num_features, segment_size)
        
        # Confidence estimation branch
        self.confidence_branch = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3),  # 3 confidence values for name, age, ID
            nn.Sigmoid()
        )
        
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, stego_image):
        # Initial processing
        x = self.conv1(stego_image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.attn1(x)
        
        # Dense blocks
        for i, (dense_block, attention) in enumerate(zip(self.dense_blocks, self.attentions)):
            x = dense_block(x)
            x = attention(x)
            if i < len(self.trans_blocks):
                x = self.trans_blocks[i](x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Extract message
        message = self.message_decoder(x)
        
        # Extract redundant segments
        boundaries, quality_scores = self.segment_extractor(x, message)
        
        # Confidence values
        confidence = self.confidence_branch(x)
        
        return message, confidence, boundaries, quality_scores
        
    def extract_with_confidence(self, stego_image):
        """Extract message with enhanced redundancy handling"""
        message, confidence, boundaries, quality_scores = self.forward(stego_image)
        
        # Process message to handle redundant segments
        # In practice, this would be done by the RedundantPatientDataProcessor
        # But we return the segments and quality info to help with decoding
        
        batch_size = message.size(0)
        segments = []
        
        for b in range(batch_size):
            batch_segments = []
            prev_boundary = 0
            
            # Extract segments based on predicted boundaries
            for i in range(len(boundaries[b])):
                if boundaries[b][i] > prev_boundary and boundaries[b][i] < self.message_length:
                    segment_end = min(boundaries[b][i].item(), self.message_length)
                    segment = message[b, prev_boundary:segment_end]
                    
                    # Only add if segment is large enough
                    if segment.size(0) > self.segment_size // 2:
                        batch_segments.append({
                            'data': segment,
                            'quality': quality_scores[b, i].item()
                        })
                    
                    prev_boundary = segment_end
            
            # Add final segment
            if prev_boundary < self.message_length:
                batch_segments.append({
                    'data': message[b, prev_boundary:],
                    'quality': quality_scores[b, -1].item()
                })
            
            segments.append(batch_segments)
        
        # Return full message and processed segments for redundancy handling
        result = {
            'message': message,
            'segments': segments,
            'confidence': {
                'name': confidence[:, 0],
                'age': confidence[:, 1],
                'id': confidence[:, 2],
                'overall': torch.mean(confidence, dim=1)
            }
        }
        
        return result