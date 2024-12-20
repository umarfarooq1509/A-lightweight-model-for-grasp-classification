import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import time
from threading import Timer
from PIL import Image

# Initialize latest_logs to avoid NameError
latest_logs = None

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Paths to your target folder and the CSV file
target_folder = r"E:\datasets\rgbd-dataset 2.0\multilabel_dataset"
csv_file_path = r"E:\datasets\rgbd-dataset 2.0\grip_labels_3.0.csv"
save_dir = os.path.join(os.path.dirname(csv_file_path), "model_outputs_2020")
os.makedirs(save_dir, exist_ok=True)

# Discord webhook URL
webhook_url = "https://discord.com/api/webhooks/1273941190416404502/l2ZJ9wV0FyyLOV1zhWxWlbr4UM5qatqCX2mtFlvn4lrsAu2EgGeXrev0OgGatRiEL2m4"

# Function to send message to Discord
def send_to_discord(content):
    data = {"content": content}
    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        print(f"Failed to send message to Discord. Status code: {response.status_code}")

# Function to extract the base name from the filename
def extract_base_name(filename):
    """
    Extracts the base name of the file before the first occurrence of "_crop",
    "_depthcrop", or "_maskcrop". This approach is more reliable if filenames
    vary in the number of underscores.
    """
    if "_crop" in filename:
        base_name = filename.split('_crop', 1)[0]
    elif "_depthcrop" in filename:
        base_name = filename.split('_depthcrop', 1)[0]
    elif "_maskcrop" in filename:
        base_name = filename.split('_maskcrop', 1)[0]
    else:
        base_name = filename.rsplit('_', 1)[0]  # fallback if no known suffix
    return base_name

# Preprocessing function using PIL
def preprocess_image(image_base_name, target_folder):
    # Construct file paths for all three types of images
    rgb_image_path = os.path.join(target_folder, f"{image_base_name}_crop_1.png")
    depth_image_path = os.path.join(target_folder, f"{image_base_name}_depthcrop.png")
    mask_image_path = os.path.join(target_folder, f"{image_base_name}_maskcrop.png")

    # Load the images
    try:
        rgb_image = Image.open(rgb_image_path).convert('L')  # Convert RGB to grayscale directly
        depth_image = Image.open(depth_image_path).convert('L')
        mask_image = Image.open(mask_image_path).convert('L')
    except FileNotFoundError as e:
        print(f"Missing file: {e.filename}. Check if the file exists and the path is correct.")
        return None

    # Resize images to 32x32
    rgb_image = rgb_image.resize((32, 32))
    depth_image = depth_image.resize((32, 32))
    mask_image = mask_image.resize((32, 32))

    # Convert images to numpy arrays
    rgb_image = np.array(rgb_image)
    depth_image = np.array(depth_image)
    mask_image = np.array(mask_image)

    # Stack the grayscale, depth, and mask images to create a 3-channel input
    stacked_image = np.stack((rgb_image, depth_image, mask_image), axis=-1)

    return stacked_image

# Load the CSV file
try:
    df = pd.read_csv(csv_file_path)
    message = "CSV file loaded successfully."
    print(message)
    send_to_discord(message)
except Exception as e:
    error_message = f"Failed to load CSV file: {str(e)}"
    print(error_message)
    send_to_discord(error_message)
    raise

# Map grip types to numeric labels
grip_type_mapping = {grip_type: idx for idx, grip_type in enumerate(df['grip_type'].unique())}
df['label'] = df['grip_type'].map(grip_type_mapping)

# Convert labels to strings
df['label'] = df['label'].astype(str)

# Split the data into training and validation sets
try:
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    message = "Data split into training and validation sets."
    print(message)
    send_to_discord(message)
except Exception as e:
    error_message = f"Failed to split data: {str(e)}"
    print(error_message)
    send_to_discord(error_message)
    raise

# Custom data generator
def custom_data_generator(df, batch_size, target_folder):
    while True:
        batch_df = df.sample(n=batch_size)
        batch_images = []
        batch_labels = []

        for index, row in batch_df.iterrows():
            image_base_name = extract_base_name(row['filename'])
            label = row['label']
            
            image = preprocess_image(image_base_name, target_folder)
            if image is not None:
                batch_images.append(image)
                batch_labels.append(int(label))
            else:
                print(f"Skipping image {image_base_name} due to missing file.")

        if not batch_images:
            print("Warning: No images were loaded in this batch. Retrying...")
            continue

        batch_images = np.array(batch_images)
        batch_labels = tf.keras.utils.to_categorical(np.array(batch_labels), num_classes=len(grip_type_mapping))

        yield batch_images, batch_labels

batch_size = 128
train_generator = custom_data_generator(train_df, batch_size, target_folder)
val_generator = custom_data_generator(val_df, batch_size, target_folder)

# Build a lightweight CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(len(grip_type_mapping), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Function to send metrics to Discord
def send_metrics_to_discord(logs):
    progress_message = (
        f"Training Update:\n"
        f"Train Loss: {logs['loss']:.4f}, Train Accuracy: {logs['accuracy']:.4f}\n"
        f"Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}"
    )
    print(progress_message)
    send_to_discord(progress_message)

# Timer to send progress updates to Discord every 5 minutes
def send_progress_update():
    global latest_logs
    if latest_logs and 'loss' in latest_logs and 'accuracy' in latest_logs:
        send_metrics_to_discord(latest_logs)
    else:
        progress_message = "Training is in progress... Metrics are not yet available."
        print(progress_message)
        send_to_discord(progress_message)
    Timer(300, send_progress_update).start()  # Send update every 5 minutes

# Callback to save the model and logs
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global latest_logs
        latest_logs = logs  # Store the latest logs globally

        # Save the model after each epoch before validation starts
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.h5')
        self.model.save(model_save_path)

        # Send a progress update with the latest logs
        send_metrics_to_discord(logs)

# Start sending progress updates
send_progress_update()

# Training the model
epochs = 5  # Adjust based on your needs

try:
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_df)//batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_df)//batch_size,
        callbacks=[CustomCallback()]
    )
except Exception as e:
    error_message = f"Training failed: {str(e)}"
    print(error_message)
    send_to_discord(error_message)
    
    # Save the model despite the exception
    emergency_model_save_path = os.path.join(save_dir, 'emergency_model_save.h5')
    model.save(emergency_model_save_path)
    emergency_save_message = f"Model saved at {emergency_model_save_path} despite training failure."
    print(emergency_save_message)
    send_to_discord(emergency_save_message)
    
    # Re-raise the exception to ensure that the error is not silently ignored
    raise

# Final evaluation after training is completed
try:
    val_loss, val_acc = model.evaluate(val_generator, steps=len(val_df)//batch_size)
    final_message = f"Final validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}"
    print(final_message)
    send_to_discord(final_message)

    # Generate and save confusion matrix and classification report
    save_metrics(epochs, model, val_generator, save_dir)
except Exception as e:
    error_message = f"Evaluation failed: {str(e)}"
    print(error_message)
    send_to_discord(error_message)
    raise

# Final saving of the trained model
final_model_save_path = os.path.join(save_dir, 'final_model.h5')
model.save(final_model_save_path)
final_save_message = f"Final model saved to {final_model_save_path}"
print(final_save_message)
send_to_discord(final_save_message)
