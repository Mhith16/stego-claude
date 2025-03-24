"""
Configuration parameters for model architecture
"""

# Feature Analysis Network
FEATURE_ANALYZER_CONFIG = {
    'in_channels': 1,  # Grayscale X-ray images
    'growth_rate': 32,
    'block_config': (6, 12, 24, 16)  # Similar to DenseNet-121
}

# Encoder Network
ENCODER_CONFIG = {
    'image_channels': 1,  # Grayscale X-ray images
    'growth_rate': 32,
    'num_dense_layers': 4
}

# Decoder Network
DECODER_CONFIG = {
    'image_channels': 1,  # Grayscale X-ray images
    'growth_rate': 32,
    'num_layers': 6,
    'message_length': 1024  # Can be adjusted based on data needs
}

# Discriminator Network
DISCRIMINATOR_CONFIG = {
    'image_channels': 1,  # Grayscale X-ray images
    'growth_rate': 32,
    'num_layers': 4
}

# Noise Layer
NOISE_LAYER_CONFIG = {
    'noise_types': ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper'],
    # Parameters for each noise type
    'dropout_prob': 0.5,
    'jpeg_quality': 50,
    'gaussian_std': 0.05,
    'blur_kernel_size': 5,
    'blur_sigma': 1.0,
    'salt_pepper_density': 0.1
}