import os
from PIL import Image
import random

# Define the target directory
target_folder = r"E:\datasets\rgbd-dataset 2.0\multilabel_dataset"

# Get a list of all files in the directory
all_files = os.listdir(target_folder)

# Filter out only the base names (without extensions) for files ending with '_crop_1.png'
crop_base_files = [file[:-11] for file in all_files if file.endswith('_crop_1.png')]

# Print the number of '_crop_1.png' files found
print(f"Number of '_crop_1.png' files found: {len(crop_base_files)}")

if len(crop_base_files) == 0:
    print("No files ending with '_crop_1.png' were found in the directory.")
else:
    # Calculate 25% of the total files
    sample_size = int(0.01 * len(crop_base_files))

    # Randomly select 25% of the base files
    sampled_base_files = random.sample(crop_base_files, sample_size)

    # Read and verify the sampled files (RGB, depth, and mask)
    for i, base_name in enumerate(sampled_base_files):
        rgb_file_path = os.path.join(target_folder, f"{base_name}_crop_1.png")
        depth_file_path = os.path.join(target_folder, f"{base_name}_depthcrop.png")
        mask_file_path = os.path.join(target_folder, f"{base_name}_maskcrop.png")

        try:
            # Verify the RGB file
            rgb_img = Image.open(rgb_file_path)
            rgb_img.verify()
            print(f"File {i+1}: {rgb_file_path} loaded successfully.")

            # Verify the depth file
            depth_img = Image.open(depth_file_path)
            depth_img.verify()
            print(f"File {i+1}: {depth_file_path} loaded successfully.")

            # Verify the mask file
            mask_img = Image.open(mask_file_path)
            mask_img.verify()
            print(f"File {i+1}: {mask_file_path} loaded successfully.")

        except Exception as e:
            print(f"File {i+1}: Error loading files for base name '{base_name}'. Error: {e}")

print(f"Tested {len(sampled_base_files)} sets of files from the dataset.")
