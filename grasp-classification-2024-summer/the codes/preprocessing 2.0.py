import os
import shutil

# Path to the main dataset folder
source_folder = r'E:\datasets\rgbd-dataset 1\rgbd-dataset'

# Dictionary to map grasp types to their respective objects
grasp_classification = {
    "Power_Grasp": ["ball", "water_bottle", "stapler", "pliers", "potato"],
    "Precision_Pinch": ["glue_stick", "scissors", "rubber_eraser", "garlic", "kleenex"],
    "Tripod_Grasp": ["mushroom", "lemon", "lime", "toothpaste", "marker", "comb", "toothbrush", "greens"],
    "Lateral_Pinch": ["binder", "plate", "dry_battery", "cereal_box", "notebook"],
    "Cylindrical_Grasp": ["flashlight", "food_can", "soda_can", "shampoo", "banana"],
    "Spherical_Grasp": ["apple", "orange", "peach", "tomato", "bell_pepper", "onion", "pear"],
    "Palmar_Neutral": ["camera", "cap", "food_box", "food_cup", "food_jar", "cereal_box"],
    "Hook_Grasp": ["pitcher", "coffee_mug", "hand_towel"],
    "Palmar_Pronated": ["lightbulb", "calculator", "food_bag", "cell_phone", "bowl", "instant_noodles", "sponge"],
    "Index_Finger_Extension": ["keyboard"]
}

# Create new folders for each grasp type
for grasp_type in grasp_classification:
    folder_path = os.path.join(source_folder, grasp_type)
    os.makedirs(folder_path, exist_ok=True)

    # Move each object to its respective grasp type folder
    for obj in grasp_classification[grasp_type]:
        src_path = os.path.join(source_folder, obj)
        dst_path = os.path.join(folder_path, obj)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: {obj} folder not found in the source directory.")

        # If the object folder was successfully moved, process its subfolders
        if os.path.exists(dst_path):
            for subfolder in os.listdir(dst_path):
                subfolder_path = os.path.join(dst_path, subfolder)

                if os.path.isdir(subfolder_path):
                    for file in os.listdir(subfolder_path):
                        file_path = os.path.join(subfolder_path, file)

                        # Define paths for moving files
                        depth_dest_path = os.path.join(folder_path, "depth_images")
                        mask_dest_path = os.path.join(folder_path, "mask_images")
                        txt_dest_path = os.path.join(folder_path, "txt_files")

                        # Create directories if they don't exist
                        os.makedirs(depth_dest_path, exist_ok=True)
                        os.makedirs(mask_dest_path, exist_ok=True)
                        os.makedirs(txt_dest_path, exist_ok=True)

                        # Move depth crop images
                        if 'depth' in file and os.path.isfile(file_path):
                            shutil.move(file_path, depth_dest_path)
                            print(f"Moved {file_path} to {depth_dest_path}")

                        # Move mask crop images
                        elif 'mask' in file and os.path.isfile(file_path):
                            shutil.move(file_path, mask_dest_path)
                            print(f"Moved {file_path} to {mask_dest_path}")

                        # Move text files
                        elif file.endswith('.txt') and os.path.isfile(file_path):
                            shutil.move(file_path, txt_dest_path)
                            print(f"Moved {file_path} to {txt_dest_path}")

print("Objects, depth crop, mask crop, and text files have been successfully organized.")
