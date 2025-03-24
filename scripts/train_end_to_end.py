import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_feature_analyzer import EnhancedFeatureAnalysisDenseNet
from models.adaptive_encoder import AdaptiveSteganographyEncoder
from models.robust_decoder import RobustSteganographyDecoder
from models.discriminator import Discriminator
from models.noise_layer import NoiseLayer
from models.perceptual_loss import EnhancedSteganographyLoss
from utils.data_loader import get_data_loaders
from utils.metrics import compute_psnr, compute_ssim, compute_bit_accuracy
from utils.error_correction import PatientDataProcessor

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Uncomment this line to help debug gradient issues if they persist
    # torch.autograd.set_detect_anomaly(True)
    
    # Create model save directory
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialize patient data processor for error correction
    patient_processor = PatientDataProcessor(max_message_length=args.message_length)
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        args.xray_dir, 
        args.label_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        image_size=(args.image_width, args.image_height)
        # We're not passing processor since the original data_loader doesn't support it
    )
    
    # Initialize models
    print("Initializing models...")
    feature_analyzer = EnhancedFeatureAnalysisDenseNet(in_channels=1, pretrained=True).to(device)
    
    # Try to load pretrained weights for medical images if available
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        feature_analyzer.load_pretrained_weights(args.pretrained_weights)
    
    encoder = AdaptiveSteganographyEncoder(
        image_channels=1, 
        embedding_mode=args.embedding_mode
    ).to(device)
    
    decoder = RobustSteganographyDecoder(
        image_channels=1, 
        message_length=args.message_length
    ).to(device)
    
    discriminator = Discriminator(image_channels=1).to(device)
    noise_layer = NoiseLayer().to(device)
    
    # Enhanced loss function
    loss_fn = EnhancedSteganographyLoss(
        lambda_image=args.lambda_image,
        lambda_message=args.lambda_message,
        lambda_adv=args.lambda_adv,
        lambda_perceptual=args.lambda_perceptual,
        lambda_ssim=args.lambda_ssim
    ).to(device)
    
    # Initialize optimizers with different learning rates for pretrained parts
    optimizer_params = [
        {'params': [p for n, p in feature_analyzer.named_parameters() 
                    if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.1},
        {'params': [p for n, p in feature_analyzer.named_parameters() 
                    if 'backbone' not in n], 'lr': args.lr}
    ]
    fa_optimizer = optim.Adam(optimizer_params, lr=args.lr)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    # Try to load checkpoints if requested
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        
        feature_analyzer.load_state_dict(checkpoint['feature_analyzer_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        fa_optimizer.load_state_dict(checkpoint['fa_optimizer_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        print(f"Resuming from epoch {start_epoch}")
    
    # Curriculum learning setup - gradually increase noise intensity
    noise_curriculum = {
        'dropout_prob': np.linspace(0.1, 0.5, args.epochs),
        'jpeg_quality': np.linspace(90, 50, args.epochs),
        'gaussian_std': np.linspace(0.01, 0.05, args.epochs),
        'blur_sigma': np.linspace(0.5, 1.5, args.epochs),
        'salt_pepper_density': np.linspace(0.05, 0.1, args.epochs)
    }
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Set curriculum parameters for this epoch
        noise_params = {
            'dropout_prob': noise_curriculum['dropout_prob'][epoch],
            'jpeg_quality': noise_curriculum['jpeg_quality'][epoch],
            'gaussian_std': noise_curriculum['gaussian_std'][epoch],
            'blur_sigma': noise_curriculum['blur_sigma'][epoch],
            'salt_pepper_density': noise_curriculum['salt_pepper_density'][epoch]
        }
        
        print(f"Noise parameters: {noise_params}")
        noise_layer.update_parameters(**noise_params)
        
        # Set models to training mode
        feature_analyzer.train()
        encoder.train()
        decoder.train()
        discriminator.train()
        
        # Training metrics
        train_metrics = {
            'disc_loss': 0.0,
            'encoder_loss': 0.0,
            'decoder_loss': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'bit_accuracy': 0.0
        }
        
        # Progress bar
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, data in enumerate(train_bar):
            images = data['image'].to(device)
            messages = data['patient_data'].to(device)
            
            # Ensure messages have the right dimensions
            if messages.dim() == 3:  # [B, 1, L]
                messages = messages.squeeze(1)  # [B, L]
            
            # ----------------------
            # Train Discriminator
            # ----------------------
            disc_optimizer.zero_grad()
            
            # Generate feature weights and stego images
            with torch.no_grad():
                feature_weights = feature_analyzer(images)
                stego_images = encoder(images, messages, feature_weights)
            
            # Real and fake predictions
            real_preds = discriminator(images)
            fake_preds = discriminator(stego_images.detach())
            
            # Calculate discriminator loss
            disc_loss_dict = loss_fn.discriminator_loss(real_preds, fake_preds)
            disc_loss = disc_loss_dict['total']
            
            disc_loss.backward()
            disc_optimizer.step()
            
            # ----------------------
            # Train Encoder & Decoder & Feature Analyzer (combined)
            # ----------------------
            # Zero all gradients
            fa_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Forward passes
            feature_weights = feature_analyzer(images)
            stego_images = encoder(images, messages, feature_weights)
            disc_preds = discriminator(stego_images)
            
            # Apply noise to stego images
            noisy_stego_images = noise_layer(stego_images)
            
            # Decode messages from noisy stego images
            decoded_output = decoder.extract_with_confidence(noisy_stego_images)
            decoded_messages = decoded_output['message']
            confidence = decoded_output['confidence']
            
            # Calculate losses
            encoder_loss_dict = loss_fn.encoder_loss(images, stego_images, disc_preds)
            encoder_loss = encoder_loss_dict['total']
            
            decoder_loss_dict = loss_fn.decoder_loss(messages, decoded_messages)
            decoder_loss = decoder_loss_dict['total']
            
            # Combined loss - this is the key change
            combined_loss = encoder_loss + decoder_loss
            
            # Single backward pass on combined loss
            combined_loss.backward()
            
            # Update all networks
            fa_optimizer.step()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # ----------------------
            # Update metrics
            # ----------------------
            train_metrics['disc_loss'] += disc_loss.item()
            train_metrics['encoder_loss'] += encoder_loss.item()
            train_metrics['decoder_loss'] += decoder_loss.item()
            
            # Calculate image quality metrics
            with torch.no_grad():
                train_metrics['psnr'] += compute_psnr(images, stego_images)
                train_metrics['ssim'] += compute_ssim(images, stego_images)
                train_metrics['bit_accuracy'] += compute_bit_accuracy(messages, decoded_messages)
            
            # Update progress bar
            train_bar.set_postfix({
                'D_loss': f"{disc_loss.item():.4f}",
                'E_loss': f"{encoder_loss.item():.4f}",
                'Dec_loss': f"{decoder_loss.item():.4f}",
                'PSNR': f"{compute_psnr(images, stego_images):.2f}",
                'Acc': f"{compute_bit_accuracy(messages, decoded_messages):.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % args.log_interval == 0:
                global_step = epoch * len(train_loader) + batch_idx
                
                # Log losses
                writer.add_scalar('Train/Discriminator_Loss', disc_loss.item(), global_step)
                writer.add_scalar('Train/Encoder_Loss/Total', encoder_loss.item(), global_step)
                writer.add_scalar('Train/Encoder_Loss/MSE', encoder_loss_dict['mse'].item(), global_step)
                writer.add_scalar('Train/Encoder_Loss/Perceptual', encoder_loss_dict['perceptual'].item(), global_step)
                writer.add_scalar('Train/Encoder_Loss/SSIM', encoder_loss_dict['ssim'].item(), global_step)
                writer.add_scalar('Train/Decoder_Loss', decoder_loss.item(), global_step)
                
                # Log metrics
                writer.add_scalar('Train/PSNR', compute_psnr(images, stego_images), global_step)
                writer.add_scalar('Train/SSIM', compute_ssim(images, stego_images), global_step)
                writer.add_scalar('Train/BitAccuracy', compute_bit_accuracy(messages, decoded_messages), global_step)
                
                # Log confidence values
                if 'overall' in confidence:
                    writer.add_scalar('Train/Confidence', confidence['overall'].mean().item(), global_step)
                
                # Log images occasionally
                if batch_idx % (args.log_interval * 10) == 0:
                    writer.add_images('Train/Original', images[:4], global_step)
                    writer.add_images('Train/Stego', stego_images[:4], global_step)
                    writer.add_images('Train/Noisy_Stego', noisy_stego_images[:4], global_step)
                    writer.add_images('Train/Feature_Weights', feature_weights[:4], global_step)
        
        # Calculate average training metrics
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        print(f"Training Metrics:")
        print(f"  Discriminator Loss: {train_metrics['disc_loss']:.6f}")
        print(f"  Encoder Loss: {train_metrics['encoder_loss']:.6f}")
        print(f"  Decoder Loss: {train_metrics['decoder_loss']:.6f}")
        print(f"  PSNR: {train_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {train_metrics['ssim']:.4f}")
        print(f"  Bit Accuracy: {train_metrics['bit_accuracy']:.4f}")
        
        # ----------------------
        # Validation
        # ----------------------
        feature_analyzer.eval()
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        
        val_metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'bit_accuracy': 0.0,
            'confidence': 0.0
        }
        
        print("Running validation...")
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                images = data['image'].to(device)
                messages = data['patient_data'].to(device)
                
                # Ensure messages have the right dimensions
                if messages.dim() == 3:
                    messages = messages.squeeze(1)
                
                # Generate stego images
                feature_weights = feature_analyzer(images)
                stego_images = encoder(images, messages, feature_weights)
                
                # Test with different noise types
                noise_types = ['dropout', 'jpeg', 'gaussian', 'blur', 'salt_pepper']
                bit_accs = []
                confidences = []
                
                for noise_type in noise_types:
                    noisy_stego = noise_layer(stego_images, noise_type)
                    decoded_output = decoder.extract_with_confidence(noisy_stego)
                    decoded_msgs = decoded_output['message']
                    confidence = decoded_output['confidence']['overall'].mean().item()
                    
                    bit_accs.append(compute_bit_accuracy(messages, decoded_msgs))
                    confidences.append(confidence)
                
                # Average metrics across noise types
                val_metrics['bit_accuracy'] += sum(bit_accs) / len(bit_accs)
                val_metrics['confidence'] += sum(confidences) / len(confidences)
                
                # Image quality metrics
                val_metrics['psnr'] += compute_psnr(images, stego_images)
                val_metrics['ssim'] += compute_ssim(images, stego_images)
        
        # Calculate average validation metrics
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Log validation metrics
        writer.add_scalar('Validation/PSNR', val_metrics['psnr'], epoch)
        writer.add_scalar('Validation/SSIM', val_metrics['ssim'], epoch)
        writer.add_scalar('Validation/BitAccuracy', val_metrics['bit_accuracy'], epoch)
        writer.add_scalar('Validation/Confidence', val_metrics['confidence'], epoch)
        
        print(f"Validation Metrics:")
        print(f"  PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {val_metrics['ssim']:.4f}")
        print(f"  Bit Accuracy: {val_metrics['bit_accuracy']:.4f}")
        print(f"  Confidence: {val_metrics['confidence']:.4f}")
        
        # Save models
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.model_save_path, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            
            # Save checkpoint for resuming
            torch.save({
                'epoch': epoch,
                'feature_analyzer_state_dict': feature_analyzer.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'fa_optimizer_state_dict': fa_optimizer.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
            }, os.path.join(save_path, 'model_checkpoint.pth'))
            
            # Save individual models for easier loading
            torch.save(feature_analyzer.state_dict(), os.path.join(save_path, 'feature_analyzer.pth'))
            torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.pth'))
            torch.save(discriminator.state_dict(), os.path.join(save_path, 'discriminator.pth'))
            
            print(f"Models saved to {save_path}")
    
    # Save final models
    final_path = os.path.join(args.model_save_path, "final_models")
    os.makedirs(final_path, exist_ok=True)
    
    torch.save(feature_analyzer.state_dict(), os.path.join(final_path, 'feature_analyzer.pth'))
    torch.save(encoder.state_dict(), os.path.join(final_path, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(final_path, 'decoder.pth'))
    torch.save(discriminator.state_dict(), os.path.join(final_path, 'discriminator.pth'))
    
    print(f"Final models saved to {final_path}")
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train enhanced steganography models")
    
    # Data paths
    parser.add_argument('--xray_dir', type=str, required=True, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing patient data text files')
    parser.add_argument('--pretrained_weights', type=str, default='', help='Path to pretrained medical model weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint for resuming training')
    
    # Model parameters
    parser.add_argument('--embedding_mode', type=str, default='spatial', choices=['spatial', 'frequency'], 
                        help='Embedding domain for encoder')
    
    # Loss weights
    parser.add_argument('--lambda_message', type=float, default=20.0, help='Weight for message loss')
    parser.add_argument('--lambda_image', type=float, default=1.0, help='Weight for image MSE loss')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Weight for adversarial loss')
    parser.add_argument('--lambda_perceptual', type=float, default=5.0, help='Weight for perceptual loss')
    parser.add_argument('--lambda_ssim', type=float, default=5.0, help='Weight for SSIM loss')
    
    # Message configuration
    parser.add_argument('--message_length', type=int, default=1024, help='Length of binary message')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')
    parser.add_argument('--model_save_path', type=str, default='./models/weights', help='Directory to save models')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging')
    parser.add_argument('--save_interval', type=int, default=5, help='How many epochs to wait before saving')
    
    # Image dimensions
    parser.add_argument('--image_width', type=int, default=256, help='Target image width')
    parser.add_argument('--image_height', type=int, default=256, help='Target image height')
    
    args = parser.parse_args()
    train(args)