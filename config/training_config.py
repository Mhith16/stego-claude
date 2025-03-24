"""
Configuration parameters for training
"""

# Basic training parameters
TRAINING_CONFIG = {
    'batch_size': 8,
    'epochs': 100,
    'learning_rate': 0.0001,
    'val_split': 0.2
}

# Loss weights
LOSS_WEIGHTS = {
    'lambda_message': 20.0,  # Higher weight for message reconstruction
    'lambda_image': 1.0,     # Weight for image distortion
    'lambda_adv': 0.1        # Weight for adversarial loss
}

# Logging and checkpoints
LOGGING_CONFIG = {
    'log_dir': './logs',
    'model_save_path': './models/weights',
    'log_interval': 10,      # Log every N batches
    'save_interval': 5       # Save models every N epochs
}

# Data processing
DATA_CONFIG = {
    'message_length': 1024,  # Length of binary message
    'image_size': (256, 256) # Target image size (width, height)
}