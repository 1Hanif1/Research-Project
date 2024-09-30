import cv2
import numpy as np
import os
import glob

# Function to extract simple features (mean pixel intensity) from the segmented lung images
def extract_features(image_path):
    # Load the segmented image (lung mask)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # If the image cannot be loaded, log an error
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Extract simple features - here, we use mean pixel intensity as a feature
    mean_intensity = np.mean(image)
    
    # You can add more complex features here (e.g., texture, shape descriptors)
    
    return [mean_intensity]

# Function to process all images and extract features
def process_images(segmented_dir):
    features = []
    print("processing Images: {}".format(segmented_dir))
    print(glob.glob(os.path.join(segmented_dir, "*.png")))

    # Process each image in the segmented directory
    for image_file in glob.glob(os.path.join(segmented_dir, "*.png")):
        print(f"Processing image: {image_file}")
        
        # Extract features
        feature_vector = extract_features(image_file)
        
        if feature_vector is not None:
            features.append(feature_vector)
    
    return np.array(features)

# Path to your segmented images
segmented_dir = r'processed_data/segmented_lungs'

if __name__ == "__main__":
    # Extract features from all images
    extracted_features = process_images(segmented_dir)

    # Output the shape of the feature matrix
    print(f"Number of images processed: {len(extracted_features)}")
    print(f"Shape of feature matrix: {extracted_features.shape}")

    # Optionally, you can save the extracted features to a file for later use
    np.save("extracted_features.npy", extracted_features)
