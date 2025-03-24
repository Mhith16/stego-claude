# extract_data.py
import os
import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.robust_decoder import RobustSteganographyDecoder
from utils.error_correction import PatientDataProcessor

def extract_patient_data(stego_path, model_path, image_size=(256, 256)):
    """
    Extract patient data from a stego image with the improved model
    
    Args:
        stego_path: Path to the stego image
        model_path: Directory containing trained models
        image_size: Input image size (width, height)
    
    Returns:
        The extracted patient data as text and confidence metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load decoder model
    decoder = RobustSteganographyDecoder(image_channels=1).to(device)
    
    # Initialize patient data processor
    processor = PatientDataProcessor()
    
    try:
        decoder_path = os.path.join(model_path, 'decoder.pth')
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.eval()
        
        # Load stego image
        stego_image = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
        if stego_image is None:
            print(f"Error: Could not read image from {stego_path}")
            return None, None
        
        # Convert to float and normalize to [0, 1]
        stego_image = stego_image.astype(np.float32) / 255.0
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])
        
        if isinstance(stego_image, np.ndarray):
            # Convert numpy array to PIL Image for torchvision transforms
            from PIL import Image
            stego_pil = Image.fromarray((stego_image * 255).astype(np.uint8))
            stego_tensor = transform(stego_pil).unsqueeze(0).to(device)
        else:
            # Already a tensor
            stego_tensor = transform(stego_image).unsqueeze(0).to(device)
        
        # Extract data with confidence values
        print("Extracting data from stego image...")
        with torch.no_grad():
            output = decoder.extract_with_confidence(stego_tensor)
            binary_output = output['message']
            confidence_scores = output['confidence']
            
            # Print confidence scores for each field
            print(f"Extraction confidence:")
            for field, value in confidence_scores.items():
                if field != 'overall':
                    print(f"  {field.capitalize()}: {value.item():.4f}")
                    
            # Overall confidence
            overall_confidence = confidence_scores['overall'].item()
            print(f"  Overall: {overall_confidence:.4f}")
        
        # Decode binary data to text
        decoded_text, field_confidence = processor.decode_patient_data(binary_output[0])
        
        print("\nExtracted patient data:")
        print("-" * 40)
        print(decoded_text)
        print("-" * 40)
        
        # Print field-specific confidence from error correction
        print("\nError correction confidence:")
        for field, value in field_confidence.items():
            if field != 'overall':
                print(f"  {field.capitalize()}: {value:.4f}")
        
        return decoded_text, {**confidence_scores, **field_confidence}
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
# Main execution for extract_data.py
if __name__ == "__main__" and os.path.basename(__file__) == "extract_data.py":
    parser = argparse.ArgumentParser(description="Extract patient data from a stego image")
    parser.add_argument('--image', type=str, required=True, help='Path to the stego image')
    parser.add_argument('--model_path', type=str, default='./models/weights/final_models', help='Path to trained models')
    parser.add_argument('--output', type=str, help='Path to save the extracted text (optional)')
    
    args = parser.parse_args()
    
    extracted_text, confidence = extract_patient_data(args.image, args.model_path)
    
    if extracted_text and args.output:
        with open(args.output, 'w') as f:
            f.write(extracted_text)
        print(f"Extracted text saved to: {args.output}")