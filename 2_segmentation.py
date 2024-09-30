import cv2
import numpy as np
import os

# Function to perform lung segmentation using Otsu's thresholding
def lung_segmentation(image_path, output_dir):
    # Load the preprocessed 2D slice (PNG format)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Perform Otsu's thresholding to segment the lung area
    _, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the thresholding result (lungs should be white)
    lung_mask = cv2.bitwise_not(thresh)
    
    # Optionally remove small contours/noise (morphological operations)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, kernel)
    
    # Save the segmented lung mask
    output_file = os.path.join(output_dir, os.path.basename(image_path).replace('.png', '_segmented.png'))
    cv2.imwrite(output_file, cleaned_mask)
    
    print(f"Segmented lung saved: {output_file}")

# Directory where your preprocessed PNG images are saved
input_dir = r'processed_data/seg-lungs-LUNA16-png'

# Directory where segmented images will be saved
output_dir = r'processed_data/segmented_lungs/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for file in os.listdir(input_dir):
    if file.endswith(".png"):
        lung_segmentation(os.path.join(input_dir, file), output_dir)
