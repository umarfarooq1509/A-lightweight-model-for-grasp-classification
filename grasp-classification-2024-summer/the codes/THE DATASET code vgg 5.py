import os
import logging
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer  # Import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

# Enable Mixed Precision Training
set_global_policy('mixed_float16')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set image size and batch size
img_size = (96, 96)
batch_size = 128  # Increased batch size for better GPU utilization

# Set the dataset path
dataset_path = r'E:\datasets\the dataset'  # Update this path

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset path {dataset_path} does not exist.")

# Custom function to preprocess images
def preprocess_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img, label

# Custom function to load dataset
def load_dataset(directory, subset, validation_split=0.3):
    filepaths, labels = [], []
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            for subfolder in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path):
                        filepath = os.path.join(subfolder_path, filename)
                        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                            filepaths.append(filepath)
                            labels.append(class_dir)
    
    # Ensure the use of LabelBinarizer for label encoding
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    num_classes = len(label_binarizer.classes_)

    # Split the data into training and validation sets
    split_idx = int(len(filepaths) * (1 - validation_split))
    if subset == 'training':
        filepaths = filepaths[:split_idx]
        labels = labels[:split_idx]
    else:
        filepaths = filepaths[split_idx:]
        labels = labels[split_idx:]

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(filepaths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, num_classes

# Load training and validation datasets
train_dataset, num_classes = load_dataset(dataset_path, subset='training')
validation_dataset, _ = load_dataset(dataset_path, subset='validation')

# Define the SmallerVGG-like model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    MaxPooling2D((3, 3)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((3, 3)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((3, 3)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', dtype='float32')  # Ensure final layer is float32
])

# Compile the model with mixed precision optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Directory to save models
checkpoint_dir = 'saved_models_of_the_dataset'
os.makedirs(checkpoint_dir, exist_ok=True)

# Checkpoint callback to save model after every epoch
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
    save_weights_only=False,
    save_freq='epoch'
)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Enable XLA Compiler
tf.config.optimizer.set_jit(True)

logger.info("Starting model training...")

# Train the model
try:
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,  # Adjust based on your needs
        callbacks=[early_stopping, checkpoint_callback]
    )
except Exception as e:
    logger.error(f"Error during training: {e}")
    raise

logger.info("Model training completed.")

# Evaluate the model on the validation set
logger.info("Evaluating model on validation set...")
val_loss, val_accuracy = model.evaluate(validation_dataset)
logger.info(f'Validation accuracy: {val_accuracy:.4f}, Validation loss: {val_loss:.4f}')

# Save the final model
model.save(os.path.join(checkpoint_dir, 'final_model.h5'))
logger.info("Final model saved as 'final_model.h5'.")

# Plotting training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()
