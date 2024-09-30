import SimpleITK as sitk
import cv2
import os
import glob

# Function to load .mhd and .raw files using SimpleITK
def load_mhd_image(image_path):
    print(f"Loading image: {image_path}")
    # Load the image using SimpleITK
    image = sitk.ReadImage(image_path)
    
    # Convert SimpleITK image to a NumPy array (3D array)
    image_array = sitk.GetArrayFromImage(image)
    
    return image_array

# Preprocess and resize images
def preprocess_mhd_images(input_dir, output_dir, image_size=(256, 256)):
    print("Starting preprocessing...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check for .mhd files
    mhd_files = glob.glob(os.path.join(input_dir, '**', '*.mhd'), recursive=True)

    if not mhd_files:
        print("No .mhd files found in the directory.")
        return
    
    # Process each .mhd file
    for file in mhd_files:
        print(f"Processing file: {file}")
        
        # Load the .mhd image
        image_array = load_mhd_image(file)

        # Each .mhd file is 3D (many slices), so process one slice at a time
        for i, slice_image in enumerate(image_array):
            # Resize the slice
            resized_image = cv2.resize(slice_image, image_size)
            
            # Save the preprocessed slice as a PNG for further use
            output_file = os.path.join(output_dir, os.path.basename(file).replace('.mhd', f'_slice_{i}.png'))
            cv2.imwrite(output_file, resized_image)
            print(f"Processed and saved: {output_file}")

if __name__ == '__main__':
    preprocess_mhd_images(r'C:\Users\Hanif Barbhuiya\OneDrive\Desktop\Research-Project\data\seg-lungs-LUNA16', r'C:\Users\Hanif Barbhuiya\OneDrive\Desktop\Research-Project\processed_data\seg-lungs-LUNA16-png')
