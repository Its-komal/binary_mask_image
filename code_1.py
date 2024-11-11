import cv2
import numpy as np
import os

# Path where all images are stored
input_folder = r'input'  
output_dir = r'mask_output'
os.makedirs(output_dir, exist_ok=True)

image_extensions = ('.jpg', '.png')
input_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(image_extensions)]

# global pixel count
total_mask_pixel_count_no_thread = 0

# Function to process images sequentially
def process_images_from_folder(image_paths):
    global total_mask_pixel_count_no_thread
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        
        # Create a binary mask where all 3 channels are above 200
        mask = np.all(image > 200, axis=-1).astype(np.uint8) * 255
        
        # Save the mask as a PNG (lossless format)
        output_filename = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_mask_no_thread.png')
        cv2.imwrite(output_filename, mask)
        
        # Count the number of pixels where the mask is max (255)
        mask_pixel_count = np.sum(mask == 255)
        total_mask_pixel_count_no_thread += mask_pixel_count

process_images_from_folder(input_paths)

print(f"Total number of pixels where the mask is max: {total_mask_pixel_count_no_thread}")
