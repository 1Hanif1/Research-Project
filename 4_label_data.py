import numpy as np
import pandas as pd
import os
import glob

# Load extracted features
features = np.load("extracted_features.npy")
print(f"Loaded {features.shape[0]} feature vectors.")

# Load annotations CSV
annotations = pd.read_csv(r'data/annotations.csv')

# Function to assign labels (same logic as before)
def get_label_for_image(image_file, annotations):
    # Extract seriesuid from the filename (assuming it is the first part of the filename)
    image_name = os.path.basename(image_file).split('_')[0]
    
    # Filter annotations for the corresponding seriesuid
    matching_annotations = annotations[annotations['seriesuid'] == image_name]

    if len(matching_annotations) == 0:
        return 0

    # Get the slice index (if available in the filename)
    try:
        slice_index = int(image_file.split('_slice_')[-1].split('_')[0])
    except ValueError:
        print(f"Warning: Unable to extract slice index from {image_file}")
        return 0
    
    # Check if the Z-coordinate matches the slice index
    for _, annotation in matching_annotations.iterrows():
        if abs(slice_index - annotation['coordZ']) <= 1:
            return 1  # Nodule found in this slice
    
    return 0  # No nodule found

# Assuming you have the list of image filenames used for feature extraction
segmented_dir = r'processed_data/segmented_lungs/'
image_files = glob.glob(os.path.join(segmented_dir, "*.png"))

# Assign labels based on annotations
labels = []
for image_file in image_files:
    label = get_label_for_image(image_file, annotations)
    labels.append(label)
    print("Labelled image file: {0}\n\n".format(image_file))

labels = np.array(labels)
print(f"Assigned labels to {len(labels)} images.")

# Save features and labels together
np.save("labeled_data.npy", {"features": features, "labels": labels})
print("Labeled data saved to labeled_data.npy")
