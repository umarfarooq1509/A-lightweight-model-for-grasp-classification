import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image, ImageFile
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the dataset path
dataset_path = r'E:\datasets\2022 paper'  # Update this path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set image size and batch size
img_size = (96, 96)
batch_size = 64  # Adjust based on your system's memory capacity

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset path {dataset_path} does not exist.")

# Image data generators with rescaling
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3  # Use 30% for validation/test split later
)

# Load images using flow_from_directory with .repeat()
logger.info("Loading dataset using flow_from_directory...")

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # 70% for training
)
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, img_size[0], img_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
    )
).repeat()  # Repeat indefinitely or specify a number of epochs

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # 20% for validation
)
validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, img_size[0], img_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, validation_generator.num_classes), dtype=tf.float32)
    )
).repeat()

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

logger.info(f"Train generator: {train_generator.samples} samples, steps per epoch: {steps_per_epoch}")
logger.info(f"Validation generator: {validation_generator.samples} samples, validation steps: {validation_steps}")

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
    Dense(5, activation='softmax')  # Change '5' to the actual number of classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Directory to save models
checkpoint_dir = 'testing 2022 paper saved models'
os.makedirs(checkpoint_dir, exist_ok=True)

# Checkpoint callback to save model after every epoch
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras'),
    save_weights_only=False,
    save_freq='epoch'
)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

logger.info("Starting model training...")

# Train the model
try:
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,  # Adjust based on your needs
        callbacks=[early_stopping, checkpoint_callback],
        steps_per_epoch=steps_per_epoch,  # Correct steps per epoch
        validation_steps=validation_steps  # Correct validation steps
    )
except Exception as e:
    logger.error(f"Error during training: {e}")
    raise

logger.info("Model training completed.")

# Evaluate the model on the test set
logger.info("Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
logger.info(f'Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}')

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
