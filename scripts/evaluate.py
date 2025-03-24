import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_analyzer import FeatureAnalysisDenseNet
from models.encoder import SteganographyEncoder
from models.decoder import SteganographyDecoder
from models.noise_layer import NoiseLayer
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_bit_accuracy, compute_embedding_capacity
from utils.data_loader import XrayDataset


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    dataset = XrayDataset(args.xray_dir, args.label_dir)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize models
    feature_analyzer = FeatureAnalysisDenseNet(in_channels=1).to(device)
    encoder = SteganographyEncoder(image_channels=1).to(device)
    decoder = SteganographyDecoder(image_channels=1, message_length=args.message_length).to(device)
    noise_layer = NoiseLayer().to(device)
    
    # Load weights
    feature_analyzer.load_state_dict(torch.load(os.path.join(args.model_path, 'feature_analyzer.pth')))
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'encoder.pth')))
    decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'decoder.pth')))
    
    # Set models to evaluation mode
    feature_analyzer.eval()
    encoder.eval()
    decoder.eval()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Evaluation metrics
    noise_types = ['none', 'dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']
    metrics = {
        noise_type: {
            'psnr': [],
            'ssim': [],
            'bit_accuracy': [],
            'embedding_capacity': []
        } for noise_type in noise_types
    }
    
    example_images = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader)):
            images = data['image'].to(device)
            messages = data['patient_data'].to(device)
            
            # Ensure messages have the right dimensions
            if messages.dim() == 3:  # [B, 1, L]
                messages = messages.squeeze(1)  # [B, L]
            
            # Generate feature weights
            feature_weights = feature_analyzer(images)
            
            # Generate stego images
            stego_images = encoder(images, messages, feature_weights)
            
            # Compute metrics for clean stego images
            metrics['none']['psnr'].append(compute_psnr(images, stego_images))
            metrics['none']['ssim'].append(compute_ssim(images, stego_images))
            
            # Extract message from clean stego images
            decoded_msgs = decoder(stego_images)
            metrics['none']['bit_accuracy'].append(compute_bit_accuracy(messages, decoded_msgs))
            
            # Compute embedding capacity
            embedding_capacity = compute_embedding_capacity(images.shape, args.message_length)
            metrics['none']['embedding_capacity'].append(embedding_capacity)
            
            # Apply different noise types and evaluate
            for noise_type in noise_types:
                if noise_type == 'none':
                    continue
                
                # Apply noise
                noisy_stego = noise_layer(stego_images, noise_type)
                
                # Decode message from noisy stego images
                decoded_msgs = decoder(noisy_stego)
                
                # Compute bit accuracy
                bit_acc = compute_bit_accuracy(messages, decoded_msgs)
                metrics[noise_type]['bit_accuracy'].append(bit_acc)
                metrics[noise_type]['embedding_capacity'].append(embedding_capacity * bit_acc)
                
                # PSNR and SSIM are the same as 'none' since we're comparing original to stego
                metrics[noise_type]['psnr'] = metrics['none']['psnr']
                metrics[noise_type]['ssim'] = metrics['none']['ssim']
            
            # Save example images from the first batch
            if batch_idx == 0:
                for i in range(min(4, images.size(0))):
                    example = {
                        'original': images[i].cpu().numpy(),
                        'stego': stego_images[i].cpu().numpy(),
                        'feature_weights': feature_weights[i].cpu().numpy(),
                        'noisy_images': {}
                    }
                    
                    for noise_type in noise_types:
                        if noise_type == 'none':
                            continue
                        noisy_stego = noise_layer(stego_images[i:i+1], noise_type)
                        example['noisy_images'][noise_type] = noisy_stego[0].cpu().numpy()
                    
                    example_images.append(example)
    
    # Compute average metrics
    avg_metrics = {
        noise_type: {
            metric: np.mean(values) for metric, values in noise_metrics.items()
        } for noise_type, noise_metrics in metrics.items()
    }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"{'Noise Type':15} {'PSNR (dB)':10} {'SSIM':10} {'Bit Accuracy':15} {'Capacity (bpp)':15}")
    print("-" * 65)
    
    for noise_type, noise_metrics in avg_metrics.items():
        print(f"{noise_type:15} {noise_metrics['psnr']:10.2f} {noise_metrics['ssim']:10.4f} {noise_metrics['bit_accuracy']:15.4f} {noise_metrics['embedding_capacity']:15.4f}")
    
    # Save results
    with open(os.path.join(args.results_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"{'Noise Type':15} {'PSNR (dB)':10} {'SSIM':10} {'Bit Accuracy':15} {'Capacity (bpp)':15}\n")
        f.write("-" * 65 + "\n")
        
        for noise_type, noise_metrics in avg_metrics.items():
            f.write(f"{noise_type:15} {noise_metrics['psnr']:10.2f} {noise_metrics['ssim']:10.4f} {noise_metrics['bit_accuracy']:15.4f} {noise_metrics['embedding_capacity']:15.4f}\n")
    
    # Generate and save visualizations
    generate_visualizations(example_images, args.results_dir)
    
    print(f"\nResults saved to {args.results_dir}")

def generate_visualizations(example_images, results_dir):
    """Generate and save visualizations of the steganography results"""
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, example in enumerate(example_images):
        # 1. Original vs Stego comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(example['original'][0], cmap='gray')
        plt.title("Original X-ray")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(example['stego'][0], cmap='gray')
        plt.title("Stego Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        diff = np.abs(example['original'][0] - example['stego'][0])
        plt.imshow(diff, cmap='hot')
        plt.title("Difference (Enhanced)")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'example_{i+1}_comparison.png'), dpi=200)
        plt.close()
        
        # 2. Feature weights visualization
        plt.figure(figsize=(6, 6))
        plt.imshow(example['feature_weights'][0], cmap='viridis')
        plt.title("Feature Density Map")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'example_{i+1}_feature_weights.png'), dpi=200)
        plt.close()
        
        # 3. Noise comparisons
        noise_types = list(example['noisy_images'].keys())
        n_cols = 3
        n_rows = (len(noise_types) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for j, noise_type in enumerate(noise_types):
            plt.subplot(n_rows, n_cols, j + 1)
            plt.imshow(example['noisy_images'][noise_type][0], cmap='gray')
            plt.title(f"Noise: {noise_type}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'example_{i+1}_noise_comparison.png'), dpi=200)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate steganography models")
    
    # Data paths
    parser.add_argument('--xray_dir', type=str, required=True, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing patient data text files')
    
    # Model paths
    parser.add_argument('--model_path', type=str, required=True, help='Directory containing trained models')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--message_length', type=int, default=1024, help='Length of binary message')
    
    # Results
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    evaluate(args)