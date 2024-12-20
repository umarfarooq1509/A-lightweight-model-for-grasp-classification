import os
import cv2
import numpy as np
import torch
from torch.multiprocessing import Pool, cpu_count
from segment_anything import SamPredictor, sam_model_registry
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load SAM2 model (ViT-B in this example)
sam_checkpoint = "E:/codes/segment-anything/sam_vit_b.pth"  # Update with your path
model_type = "vit_b"  # Can be 'vit_b', 'vit_l', or 'vit_h'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Move model to GPU
sam.to(device)

# Paths
dataset_dir = "E:/datasets/2022 paper/"
mask_save_dir = "E:/datasets/2022 paper masks/"

# Create the mask_save_dir if it doesn't exist
os.makedirs(mask_save_dir, exist_ok=True)

# Function to generate and save masks
def process_image(image_info):
    image_path, mask_save_path = image_info
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to PyTorch tensor and move it to the GPU
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).to(device)

        # Generate segmentation mask using SAM2
        predictor.set_image(image_tensor)
        mask = predictor.predict()[0]

        # Convert mask to numpy array if it's a tensor
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # Ensure the mask is 2D by taking the first channel if needed
        if mask.ndim == 3:
            mask = mask[0]  # Select the first channel if mask is 3D

        # Save the mask
        cv2.imwrite(mask_save_path, mask * 255)  # Convert mask to binary (0 or 255)

        logger.info(f"Saved mask: {mask_save_path}")
    
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")

# Function to create the image_info list
def get_image_info_list(dataset_dir, mask_save_dir):
    image_info_list = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, dataset_dir)
                mask_save_path = os.path.join(mask_save_dir, relative_path)

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)

                image_info_list.append((image_path, mask_save_path))
    
    return image_info_list

# Generate the image_info_list
image_info_list = get_image_info_list(dataset_dir, mask_save_dir)

# Reduce the number of CPU workers to lower CPU and RAM usage
num_workers = max(cpu_count() // 2, 1)

# Use multiprocessing to speed up mask generation
if __name__ == '__main__':
    with Pool(processes=num_workers) as pool:
        pool.map(process_image, image_info_list)

    logger.info("Finished generating masks for all images.")
