import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

# Ensure truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224  # VGG16 expects 224x224 images
BATCH_SIZE = 64
EPOCHS = 20
NUM_CLASSES = 3
DATA_DIR = "E:\\datasets"

# Set intra-op and inter-op parallelism threads
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def safe_load_img(path, color_mode='rgb', target_size=(224, 224)):
    try:
        if path.lower().endswith('.eps'):
            print(f"Skipping unsupported file format: {path}")
            return None
        img = load_img(path, color_mode=color_mode, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def image_data_generator(directory, target_size=(224, 224), color_mode='rgb'):
    class_names = sorted(os.listdir(directory))
    while True:
        X, y = [], []
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(directory, class_name)
            filenames = os.listdir(class_dir)
            random.shuffle(filenames)
            for fname in filenames:
                fpath = os.path.join(class_dir, fname)
                img = safe_load_img(fpath, color_mode=color_mode, target_size=target_size)
                if img is not None:
                    X.append(img)
                    y.append(label)
                    if len(X) == BATCH_SIZE:
                        X = np.array(X)
                        y = np.array(y)
                        y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
                        yield X, y
                        X, y = [], []

# Create TensorFlow datasets
train_gen = lambda: image_data_generator(DATA_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH))
train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature=(tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Model Building with VGG16 for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers on top of VGG16
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  # Ensure output layer has the correct number of units
])

# Model Compilation with a reduced learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='TF_again_{epoch:02d}.h5',
    save_weights_only=False,
    save_freq='epoch'
)

# Train the model
print("Starting training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=len(os.listdir(DATA_DIR)) * 100 // BATCH_SIZE,  # Assuming 100 images per class
    callbacks=[checkpoint_callback]
)
