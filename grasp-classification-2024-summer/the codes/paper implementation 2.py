import os
from PIL import Image, ImageFile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.ndimage import gaussian_filter, median_filter

# Ensure truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Disable GPU acceleration and use only CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Constants
IMG_HEIGHT, IMG_WIDTH = 48, 36
BATCH_SIZE = 64
EPOCHS = 5
NUM_CLASSES = 5  # Update this to the correct number of classes

# Data Path (combined directory)
DATA_DIR = "E:\\datasets"

def safe_load_img(path, color_mode='grayscale', target_size=(48, 36)):
    try:
        img = load_img(path, color_mode=color_mode, target_size=target_size)
        img_array = img_to_array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def preprocess_image(img_array):
    img_array = gaussian_filter(img_array, sigma=1)
    img_array = median_filter(img_array, size=3)
    img_mean = np.mean(img_array)
    img_std = np.std(img_array)
    img_array = (img_array - img_mean) / img_std
    return img_array

def generator_with_safety(datagen, directory, target_size, batch_size, class_mode, color_mode, subset):
    gen = datagen.flow_from_directory(directory, target_size=target_size, batch_size=batch_size, class_mode=class_mode, color_mode=color_mode, subset=subset)
    while True:
        batch_x, batch_y = next(gen)
        safe_batch_x = []
        safe_batch_y = []
        for i in range(len(batch_x)):
            img_path = gen.filepaths[gen.index_array[i]]
            img = safe_load_img(img_path, color_mode=color_mode, target_size=target_size)
            if img is not None:
                img = preprocess_image(img)
                safe_batch_x.append(img)
                safe_batch_y.append(batch_y[i])
        safe_batch_x = np.array(safe_batch_x)
        safe_batch_y = np.array(safe_batch_y)
        yield safe_batch_x, safe_batch_y

# Data Preparation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2  # 20% of data for validation
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training'
)

validation_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation'
)

train_generator = generator_with_safety(
    datagen,
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training'  # Set as training data
)

validation_generator = generator_with_safety(
    datagen,
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation'  # Set as validation data
)

# Load the saved model
model = load_model('grasp_classification_model_epoch_02.h5')

# Optionally modify the model
# (Modifying the structure requires more advanced handling of weights and should be done with care)
# Example: Add a new layer
# new_layer = Dense(128, activation='relu')
# model.add(new_layer)

# Compile the model with a new learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define the ModelCheckpoint callback to save the model after further training
checkpoint_callback = ModelCheckpoint(
    filepath='grasp_classification_model_further_epoch_{epoch:02d}.h5',
    save_weights_only=False,
    save_freq='epoch'
)

# Continue training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
)

# Model Evaluation
loss, accuracy = model.evaluate(validation_generator, steps=validation_gen.samples // BATCH_SIZE)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Verify CPU Availability (since we're not using GPU)
print("Running on CPU")
