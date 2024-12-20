import os
import csv

# Define the final grip type distribution based on object names
grip_type_distribution = {
    "Cylindrical Grip": ["water_bottle", "flashlight", "food_can", "soda_can", "shampoo", "food_jar", "bowl", "food_cup"],
    "Power Grip": ["stapler", "pliers", "binder", "cereal_box", "notebook", "food_box"],
    "Pinch Grip": ["scissors", "rubber_eraser", "garlic", "mushroom", "toothpaste", "dry_battery", "keyboard"],
    "Tripod Grip": ["toothbrush", "pen", "pencil", "glue_stick", "marker"],
    "Palmar Grip": ["kleenex", "sponge", "plate", "calculator", "cell_phone", "lightbulb", "banana", "instant_noodles", "camera", "food_bag", "greens"],
    "Spherical Grip": ["ball", "apple", "orange", "peach", "tomato", "bell_pepper", "lemon", "lime", "onion", "pear", "potato"],
    "Hook Grip": ["comb", "cap", "hand_towel", "pitcher", "coffee_mug"]
}

# Path to your target folder and output CSV
target_folder = r"E:\datasets\rgbd-dataset 2.0\multilabel_dataset"  # Update this with your actual folder path
output_csv = r"E:\datasets\rgbd-dataset 2.0\grip_labels_3.0.csv"

# Create a CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'grip_type'])  # Header row

    # Iterate through each file in the target folder
    for filename in os.listdir(target_folder):
        file_path = os.path.join(target_folder, filename)
        
        if os.path.isfile(file_path):
            # Determine the grip type based on the object name in the filename
            file_grip_type = None
            for grip_type, objects in grip_type_distribution.items():
                if any(obj in filename.lower() for obj in objects):
                    file_grip_type = grip_type
                    break
            
            # Write the filename and corresponding grip type to the CSV
            if file_grip_type:
                writer.writerow([filename, file_grip_type])

print(f"CSV file created successfully at: {output_csv}")
