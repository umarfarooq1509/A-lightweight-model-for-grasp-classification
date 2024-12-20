import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import traceback

# Discord Webhook URL
WEBHOOK_URL = 'https://discord.com/api/webhooks/1273941190416404502/l2ZJ9wV0FyyLOV1zhWxWlbr4UM5qatqCX2mtFlvn4lrsAu2EgGeXrev0OgGatRiEL2m4'

# Image dimensions
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Paths
train_dir = r"E:\datasets\rgbd-dataset 2.0\multilabel_dataset"
csv_file_path = r"E:\datasets\rgbd-dataset 2.0\grip_labels.csv"
model_save_path = r"E:\datasets\rgbd-dataset 2.0\model"

# Number of grip types (classes)
num_classes = 11

# Hyperparameters
batch_size = 128
epochs = 20
learning_rate = 0.0001

# Configure TensorFlow to reduce CPU usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Load the dataset and split into training and testing sets (70-30 split)
df = pd.read_csv(csv_file_path)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# Data Augmentation and Normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Reduced range to lower CPU load
    width_shift_range=0.1,  # Reduced range
    height_shift_range=0.1,  # Reduced range
    shear_range=0.1,  # Reduced range
    zoom_range=0.1,  # Reduced range
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='Image Name',
    y_col=['Cylindrical Grip', 'Power Grip', 'Precision Grip', 'Tripod Grip', 
           'Pinch Grip', 'Lateral Pinch', 'Hook Grip', 'Palmar Grip', 
           'Index Finger Extension Grip', 'Spherical Grip', 'Lumbrical Grip'],
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=batch_size,
    class_mode='raw',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col='Image Name',
    y_col=['Cylindrical Grip', 'Power Grip', 'Precision Grip', 'Tripod Grip', 
           'Pinch Grip', 'Lateral Pinch', 'Hook Grip', 'Palmar Grip', 
           'Index Finger Extension Grip', 'Spherical Grip', 'Lumbrical Grip'],
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

# Load the VGG19 model without the top layers
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the base model
base_model.trainable = False

# Add custom top layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callback to save the model after each epoch
checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path, 'vgg19_model_{epoch:02d}.keras'),
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min',
                             verbose=1)

# Custom callback to send metrics to Discord and print progress
class ProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            # Send metrics to Discord
            message = f"Epoch {epoch + 1}/{epochs}\n"
            message += f"Loss: {logs['loss']:.4f}\n"
            message += f"Accuracy: {logs['accuracy']:.4f}\n"
            message += f"Val Loss: {logs['val_loss']:.4f}\n"
            message += f"Val Accuracy: {logs['val_accuracy']:.4f}"
            requests.post(WEBHOOK_URL, json={'content': message})
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f" - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
            print(f" - Val Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")
            
            # Evaluate on validation data and save confusion matrix and classification report
            val_pred = np.round(model.predict(val_generator))
            y_true = val_generator.labels
            cm = confusion_matrix(y_true.argmax(axis=1), val_pred.argmax(axis=1))
            cr = classification_report(y_true.argmax(axis=1), val_pred.argmax(axis=1), target_names=train_generator.class_indices.keys())
            
            # Save confusion matrix and classification report
            cm_save_path = os.path.join(model_save_path, f'confusion_matrix_epoch_{epoch + 1}.png')
            cr_save_path = os.path.join(model_save_path, f'classification_report_epoch_{epoch + 1}.txt')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(cm_save_path)
            plt.close()
            
            with open(cr_save_path, 'w') as f:
                f.write(cr)
        except Exception as e:
            # Send exception report to Discord
            exception_message = f"Exception occurred during epoch {epoch + 1}:\n{str(e)}\n{traceback.format_exc()}"
            requests.post(WEBHOOK_URL, json={'content': exception_message})
            print(exception_message)

# Main training loop with exception handling
try:
    # Fit the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, ProgressCallback()],
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    # Final evaluation
    eval_metrics = model.evaluate(val_generator)
    print(f"Final evaluation metrics: {eval_metrics}")

    # Optionally, send final results to Discord
    requests.post(WEBHOOK_URL, json={'content': f"Training completed. Final evaluation metrics: {eval_metrics}"})

except Exception as e:
    # Send exception report to Discord if the script fails
    exception_message = f"Exception occurred:\n{str(e)}\n{traceback.format_exc()}"
    requests.post(WEBHOOK_URL, json={'content': exception_message})
    print(exception_message)
