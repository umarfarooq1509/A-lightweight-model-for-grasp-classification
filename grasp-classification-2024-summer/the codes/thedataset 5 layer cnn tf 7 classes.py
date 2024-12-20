import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split  # Import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import time
from threading import Timer

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Paths to your target folder and the CSV file
target_folder = r"E:\datasets\rgbd-dataset 2.0\multilabel_dataset"  # Update with your actual folder path
csv_file_path = r"E:\datasets\rgbd-dataset 2.0\grip_labels_2.0.csv"  # Update with your CSV file path
save_dir = os.path.join(os.path.dirname(csv_file_path), "model_outputs")
os.makedirs(save_dir, exist_ok=True)

# Discord webhook URL
webhook_url = "https://discord.com/api/webhooks/1273941190416404502/l2ZJ9wV0FyyLOV1zhWxWlbr4UM5qatqCX2mtFlvn4lrsAu2EgGeXrev0OgGatRiEL2m4"

# Function to send message to Discord
def send_to_discord(content):
    data = {"content": content}
    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        print(f"Failed to send message to Discord. Status code: {response.status_code}")

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

# Image data generator for augmentation and normalization
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Function to create data generators
def create_data_generator(df, datagen, target_folder):
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=target_folder,
        x_col='filename',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

train_generator = create_data_generator(train_df, train_datagen, target_folder)
val_generator = create_data_generator(val_df, val_datagen, target_folder)

# Build a robust CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(grip_type_mapping), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Function to save confusion matrix and classification report
def save_metrics(epoch, model, val_generator, save_dir):
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    labels = list(grip_type_mapping.keys())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Save confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
    plt.close()

    # Save classification report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, f'classification_report_epoch_{epoch+1}.csv'))

# Callback to save the model and metrics after each epoch
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.h5')
        self.model.save(model_save_path)
        save_metrics(epoch, self.model, val_generator, save_dir)
        progress_message = (
            f"Epoch {epoch+1} completed. Model saved to {model_save_path}.\n"
            f"Train Loss: {logs['loss']:.4f}, Train Accuracy: {logs['accuracy']:.4f}\n"
            f"Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}"
        )
        print(progress_message)
        send_to_discord(progress_message)

# Timer to send progress updates to Discord every 5 minutes
def send_progress_update():
    progress_message = "Training is in progress..."
    print(progress_message)
    send_to_discord(progress_message)
    Timer(300, send_progress_update).start()  # Send update every 5 minutes

# Start sending progress updates
send_progress_update()

# Training the model
epochs = 10  # Adjust based on your needs

try:
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[CustomCallback()]
    )
except Exception as e:
    error_message = f"Training failed: {str(e)}"
    print(error_message)
    send_to_discord(error_message)
    raise

# Final evaluation
val_loss, val_acc = model.evaluate(val_generator)
final_message = f"Final validation accuracy: {val_acc:.4f}"
print(final_message)
send_to_discord(final_message)

# Final saving of the model
final_model_save_path = os.path.join(save_dir, 'final_model.h5')
model.save(final_model_save_path)
final_save_message = f"Final model saved to {final_model_save_path}"
print(final_save_message)
send_to_discord(final_save_message)
