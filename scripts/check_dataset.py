# scripts/check_dataset.py
import os
import sys
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import XrayDataset

def main(args):
    # Create dataset
    dataset = XrayDataset(args.xray_dir, args.label_dir)
    
    # Check if dataset is valid
    if len(dataset) == 0:
        print("ERROR: No valid samples found!")
        return
    
    print(f"Dataset contains {len(dataset)} valid samples")
    
    # Print a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Image file: {sample['file_name']}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Patient data length: {len(sample['patient_text'])} characters")
        print(f"  Binary data shape: {sample['patient_data'].shape}")
        # Print first 50 chars of patient data
        print(f"  Patient data preview: {sample['patient_text'][:50]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset validity")
    parser.add_argument('--xray_dir', type=str, required=True, help='Directory containing X-ray images')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing patient data text files')
    
    args = parser.parse_args()
    main(args)