import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def mask_patient_data(image_path, save_path=None):
    """
    Detect and mask text (patient data) in X-ray images
    Args:
        image_path: Path to the original X-ray image
        save_path: Path to save the masked image (if None, doesn't save)
    Returns:
        The masked image as a numpy array
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Run Tesseract OCR
    ocr_results = pytesseract.image_to_data(img, output_type=Output.DICT)
    
    # Process text
    processed_text = []
    for text in ocr_results["text"]:
        # If text is empty, append a newline, else nothing
        processed_text.append(text+' ' if text.strip() else '\n')
    
    # Join and split the processed text into non-empty lines
    final_text = ''.join(processed_text).strip()
    final_text = '\n'.join([line for line in final_text.split('\n') if line])
    final_text_list = final_text.split('\n')
    
    # Identify text to remove
    remove = []
    for i in final_text_list:
        if 'Name' in i or 'Age' in i:
            remove += (i.split(' '))
    remove = [text for text in remove if text.strip() != '']
    
    # Get text regions to mask
    detected_texts = []
    for i in range(len(ocr_results["text"])):
        text = ocr_results["text"][i]
        if text.strip():  # Skip empty detections
            detected_texts.append({
                'text': text,
                'x': ocr_results['left'][i],
                'y': ocr_results['top'][i],
                'width': ocr_results['width'][i],
                'height': ocr_results['height'][i]
            })
    
    # Create a copy of the image to mask
    masked_img = img.copy()
    
    # Mask text by drawing black rectangles
    for detection in detected_texts:
        if any(term in detection['text'] for term in remove):
            # Draw a black rectangle over the text
            cv2.rectangle(
                masked_img,
                (detection['x'], detection['y']),
                (detection['x'] + detection['width'], detection['y'] + detection['height']),
                (0, 0, 0),
                -1  # Fill the rectangle
            )
    
    # Save the masked image if a save path is provided
    if save_path:
        cv2.imwrite(save_path, masked_img)
    
    return masked_img

def preprocess_dataset(raw_dir, output_dir):
    """
    Preprocess a directory of X-ray images by masking patient data
    Args:
        raw_dir: Directory containing original X-ray images
        output_dir: Directory to save masked images
    """
    import os
    from tqdm import tqdm
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(raw_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for img_file in tqdm(image_files):
        input_path = os.path.join(raw_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        # Mask patient data
        mask_patient_data(input_path, output_path)
    
    print(f"Preprocessing complete. Masked images saved to {output_dir}")