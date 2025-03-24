# embed_data.py
import os
import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_feature_analyzer import EnhancedFeatureAnalysisDenseNet
from models.adaptive_encoder import AdaptiveSteganographyEncoder
from utils.error_correction import PatientDataProcessor

def embed_patient_data(image_path, text, model_path, output_path, image_size=(256, 256), embedding_mode='spatial'):
    """
    Embed patient data into an X-ray image with the improved model
    
    Args:
        image_path: Path to the X-ray image
        text: Patient data text or path to text file
        model_path: Directory containing trained models
        output_path: Path to save the stego image
        image_size: Target image size (width, height)
        embedding_mode: 'spatial' or 'frequency' domain embedding
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    feature_analyzer = EnhancedFeatureAnalysisDenseNet(in_channels=1).to(device)
    encoder = AdaptiveSteganographyEncoder(image_channels=1, embedding_mode=embedding_mode).to(device)
    
    # Initialize patient data processor
    processor = PatientDataProcessor()
    
    try:
        print(f"Loading models from: {model_path}")
        feature_analyzer_path = os.path.join(model_path, 'feature_analyzer.pth')
        encoder_path = os.path.join(model_path, 'encoder.pth')
        
        # Load state dictionaries
        feature_analyzer.load_state_dict(torch.load(feature_analyzer_path, map_location=device))
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        
        feature_analyzer.eval()
        encoder.eval()
        
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])
        
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image for torchvision transforms
            from PIL import Image
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_tensor = transform(image_pil).unsqueeze(0).to(device)
        else:
            # Already a tensor
            image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Process patient data
        print("Processing patient data...")
        if os.path.exists(text):
            with open(text, 'r') as f:
                patient_text = f.read().strip()
        else:
            patient_text = text
            
        # Encode data with error correction
        binary_data = processor.encode_patient_data(patient_text).unsqueeze(0).to(device)
        
        # Generate stego image
        print("Generating stego image...")
        with torch.no_grad():
            # Generate feature weights
            feature_weights = feature_analyzer(image_tensor)
            
            # Create visualization of features
            feature_vis = feature_weights[0, 0].cpu().numpy()
            feature_vis = (feature_vis * 255).astype(np.uint8)
            cv2.imwrite(os.path.splitext(output_path)[0] + "_features.png", feature_vis)
            
            # Embed data
            stego_image = encoder(image_tensor, binary_data, feature_weights)
        
        # Convert to numpy and save
        stego_np = stego_image[0, 0].cpu().numpy() * 255
        stego_np = np.clip(stego_np, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, stego_np)
        
        # Calculate image quality metrics
        with torch.no_grad():
            from utils.metrics import compute_psnr, compute_ssim
            psnr = compute_psnr(image_tensor, stego_image)
            ssim = compute_ssim(image_tensor, stego_image)
            
        print(f"Stego image created successfully: {output_path}")
        print(f"Image quality: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
        
        # Create a difference image to visualize changes
        diff = np.abs(image_tensor[0, 0].cpu().numpy() - stego_image[0, 0].cpu().numpy()) * 10  # Amplify differences
        diff = (diff * 255).astype(np.uint8)
        cv2.imwrite(os.path.splitext(output_path)[0] + "_diff.png", diff)
        
        return output_path
    
    except Exception as e:
        print(f"Error embedding data: {e}")
        import traceback
        traceback.print_exc()
        return None




# Main execution for embed_data.py
if __name__ == "__main__" and os.path.basename(__file__) == "embed_data.py":
    parser = argparse.ArgumentParser(description="Embed patient data into an X-ray image")
    parser.add_argument('--image', type=str, required=True, help='Path to the X-ray image')
    parser.add_argument('--text', type=str, required=True, help='Patient data text or path to text file')
    parser.add_argument('--model_path', type=str, default='./models/weights/final_models', help='Path to trained models')
    parser.add_argument('--output', type=str, default='./stego_image.png', help='Path to save the stego image')
    parser.add_argument('--embedding_mode', type=str, default='spatial', choices=['spatial', 'frequency'], 
                        help='Embedding domain (spatial or frequency)')
    
    args = parser.parse_args()
    embed_patient_data(args.image, args.text, args.model_path, args.output, embedding_mode=args.embedding_mode)

