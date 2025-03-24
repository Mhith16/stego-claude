import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

class XrayDataset(Dataset):
    """
    Dataset for X-ray images and patient data text files
    """
    def __init__(self, xray_dir, label_dir, transform=None, image_size=(256, 256)):
        """
        Args:
            xray_dir (str): Directory with X-ray images
            label_dir (str): Directory with patient data text files
            transform (callable, optional): Optional transform to be applied on the images
            image_size (tuple): Target image size (width, height)
        """
        self.xray_dir = xray_dir
        self.label_dir = label_dir
        self.image_size = image_size
        
        # Default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # torchxrayvision expects 224x224
                transforms.ToTensor(),  # Scales to [0, 1]
            ])
        else:
            self.transform = transform
        
        # Get list of files
        self.xray_files = sorted([f for f in os.listdir(xray_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        
        # Create mapping from base filename to full filename
        xray_map = {os.path.splitext(f)[0]: f for f in self.xray_files}
        label_map = {os.path.splitext(f)[0]: f for f in self.label_files}
        
        # Find common basenames
        common_names = set(xray_map.keys()) & set(label_map.keys())
        
        # Create paired files
        self.paired_files = [(xray_map[name], label_map[name]) for name in common_names]
        
        print(f"Found {len(self.paired_files)} matching X-ray and label pairs")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        # Load X-ray image
        xray_file, label_file = self.paired_files[idx]
        img_path = os.path.join(self.xray_dir, xray_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations to ensure consistent size
        if self.transform:
            image = self.transform(image)
        
        # Load patient data
        label_path = os.path.join(self.label_dir, label_file)
        with open(label_path, 'r') as f:
            patient_data = f.read().strip()
        
        # Convert patient data to binary representation
        binary_data = self._text_to_binary(patient_data)
        
        return {
            'image': image,
            'patient_data': binary_data,
            'patient_text': patient_data,  # Keep the text for evaluation
            'file_name': xray_file
        }
    
    def _text_to_binary(self, text):
        """Convert text to binary tensor representation"""
        # Convert each character to its ASCII binary representation
        binary = ''.join([format(ord(c), '08b') for c in text])
        
        # Convert to tensor
        binary_tensor = torch.tensor([int(bit) for bit in binary], dtype=torch.float32)
        
        # For now, using a fixed size tensor - we'll later handle variable lengths properly
        max_length = 1024  # This will be adjusted based on your requirements
        
        if len(binary_tensor) > max_length:
            binary_tensor = binary_tensor[:max_length]
        else:
            # Pad with zeros if needed
            padding = torch.zeros(max_length - len(binary_tensor))
            binary_tensor = torch.cat([binary_tensor, padding])
        
        return binary_tensor

def get_data_loaders(xray_dir, label_dir, batch_size=4, transform=None, val_split=0.2, image_size=(256, 256)):
    """Create train and validation data loaders"""
    # Debug info
    print(f"Loading data from:")
    print(f"  X-ray directory: {xray_dir}")
    print(f"  Label directory: {label_dir}")
    
    # Check if directories exist
    if not os.path.exists(xray_dir):
        raise FileNotFoundError(f"X-ray directory not found: {xray_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    # List files
    xray_files = sorted([f for f in os.listdir(xray_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    
    print(f"Found {len(xray_files)} X-ray images and {len(label_files)} label files")
    
    # Create dataset
    dataset = XrayDataset(xray_dir, label_dir, transform, image_size)
    
    print(f"Dataset contains {len(dataset)} valid samples")
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found. Please check that X-ray and label files have matching names (without extension).")
    
    # Split into train/validation
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Ensure non-empty splits
    if train_size == 0 or val_size == 0:
        # If dataset is too small, use the same data for both
        train_dataset = dataset
        val_dataset = dataset
        print("Warning: Dataset too small for splitting, using same data for train and validation")
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Created train dataset with {len(train_dataset)} samples and validation dataset with {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def adapt_message_size(message, target_size):
    """Adjust message size to match target_size by padding or truncating"""
    current_size = message.size(1) if len(message.size()) > 1 else message.size(0)
    
    if current_size == target_size:
        return message
    
    # Handle batched and non-batched messages
    if len(message.size()) > 1:
        if current_size < target_size:
            # Pad with zeros
            padding = torch.zeros(message.size(0), target_size - current_size, 
                                 device=message.device)
            return torch.cat([message, padding], dim=1)
        else:
            # Truncate
            return message[:, :target_size]
    else:
        if current_size < target_size:
            # Pad with zeros
            padding = torch.zeros(target_size - current_size, device=message.device)
            return torch.cat([message, padding], dim=0)
        else:
            # Truncate
            return message[:target_size]