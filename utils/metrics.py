import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index between two images"""
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # If batched, compute average SSIM
    if img1.ndim == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            if img1.shape[1] == 1:  # Grayscale
                ssim_val = ssim(img1[i, 0], img2[i, 0], data_range=1.0)
            else:  # RGB
                ssim_val = ssim(img1[i].transpose(1, 2, 0), img2[i].transpose(1, 2, 0), 
                                multichannel=True, data_range=1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        # Single image case
        if img1.shape[0] == 1:  # Grayscale
            return ssim(img1[0], img2[0], data_range=1.0)
        else:  # RGB
            return ssim(img1.transpose(1, 2, 0), img2.transpose(1, 2, 0), 
                        multichannel=True, data_range=1.0)

def compute_bit_accuracy(original_msg, decoded_msg, threshold=0.5):
    """Compute bit accuracy between original and decoded messages"""
    if torch.is_tensor(original_msg):
        original_bits = (original_msg > threshold).float()
    else:
        original_bits = (np.array(original_msg) > threshold).astype(float)
        
    if torch.is_tensor(decoded_msg):
        decoded_bits = (decoded_msg > threshold).float()
    else:
        decoded_bits = (np.array(decoded_msg) > threshold).astype(float)
    
    # Handle size mismatch
    if torch.is_tensor(original_bits) and torch.is_tensor(decoded_bits):
        # Adjust sizes if they don't match
        if original_bits.size() != decoded_bits.size():
            if len(original_bits.size()) > 1 and len(decoded_bits.size()) > 1:
                if original_bits.size(1) < decoded_bits.size(1):
                    # Truncate decoded bits to match original size
                    decoded_bits = decoded_bits[:, :original_bits.size(1)]
                else:
                    # Pad original bits with zeros to match decoded size
                    padding = torch.zeros(original_bits.size(0), 
                                         decoded_bits.size(1) - original_bits.size(1),
                                         device=original_bits.device)
                    original_bits = torch.cat([original_bits, padding], dim=1)
        
        return torch.mean((original_bits == decoded_bits).float()).item()
    else:
        # Handle numpy arrays similarly
        if len(original_bits) != len(decoded_bits):
            min_len = min(len(original_bits), len(decoded_bits))
            original_bits = original_bits[:min_len]
            decoded_bits = decoded_bits[:min_len]
        
        return np.mean(original_bits == decoded_bits)

def compute_embedding_capacity(image_shape, message_length):
    """Compute bits per pixel (bpp) embedding capacity"""
    h, w = image_shape[-2:]
    total_pixels = h * w
    bits_per_pixel = message_length / total_pixels
    return bits_per_pixel

def compute_decoding_accuracy(original_text, decoded_text):
    """Compute character-level accuracy between original and decoded text"""
    # This will be implemented once we have text decoding
    pass