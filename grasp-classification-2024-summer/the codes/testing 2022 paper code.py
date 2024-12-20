import os
import logging
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to evaluate the model
def evaluate_model(dataset_path, model_path, img_size=(96, 96), batch_size=32, test_split=0.1):
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist.")
    
    # Load the model
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)
    logger.info(f"Model loaded successfully.")

    # Image data generator for test data
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=test_split  # Use a small portion (10%) of the data for testing
    )

    # Create test data generator
    test_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # 10% for testing
        shuffle=False  # Don't shuffle so we can map predictions correctly
    )

    # Evaluate the model on the test data
    logger.info("Evaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
    logger.info(f'Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}')

    # Get true labels and predicted labels
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, steps=len(test_generator))
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate precision, recall, F1-score
    report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
    logger.info("Classification Report:\n" + report)

    # Print accuracy
    accuracy = accuracy_score(y_true, y_pred_classes)
    logger.info(f'Overall Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_labels = list(test_generator.class_indices.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Return the accuracy, classification report, and confusion matrix
    return accuracy, report, cm

# Example usage
if __name__ == "__main__":
    dataset_path = r'E:\datasets\2022 paper'  # Update this path to your dataset
    model_path = r'E:\codes\paper implemetation\saved_models\model_epoch_15.keras'  # Update this path to your saved model
    evaluate_model(dataset_path, model_path, img_size=(96, 96), batch_size=32, test_split=0.1)
