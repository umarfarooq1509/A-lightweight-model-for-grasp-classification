import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, 
                             precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import itertools

# Increase font sizes for readability
plt.rcParams.update({
    'font.size': 16,          # Base font size
    'axes.titlesize': 18,     # Font size for axes titles
    'axes.labelsize': 16,     # Font size for x and y labels
    'xtick.labelsize': 14,    # Font size for x tick labels
    'ytick.labelsize': 14,    # Font size for y tick labels
    'legend.fontsize': 16,    # Font size for legends
})

# Paths to your target folder and the CSV file
target_folder = r"E:\datasets\rgbd-dataset 2.0\multilabel_dataset"
csv_file_path = r"E:\datasets\rgbd-dataset 2.0\grip_labels_3.0.csv"
save_dir = os.path.join(os.path.dirname(csv_file_path), "model_outputs_2020")
final_model_save_path = os.path.join(save_dir, 'model_epoch_5.h5')
results_dir = r"E:\codes\results storage"
os.makedirs(results_dir, exist_ok=True)

# Load the model
if os.path.exists(final_model_save_path):
    model = tf.keras.models.load_model(final_model_save_path)
    print(f"Loaded model from {final_model_save_path}")
else:
    raise FileNotFoundError(f"No saved model found at {final_model_save_path}")

# Function to extract the base name from the filename
def extract_base_name(filename):
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
    rgb_image_path = os.path.join(target_folder, f"{image_base_name}_crop_1.png")
    depth_image_path = os.path.join(target_folder, f"{image_base_name}_depthcrop.png")
    mask_image_path = os.path.join(target_folder, f"{image_base_name}_maskcrop.png")

    try:
        rgb_image = Image.open(rgb_image_path).convert('L')  # Convert to grayscale if needed
        depth_image = Image.open(depth_image_path).convert('L')
        mask_image = Image.open(mask_image_path).convert('L')
    except FileNotFoundError as e:
        print(f"Missing file: {e.filename}. Check if the file exists and the path is correct.")
        return None

    # Resize images to 32x32 (as required by the model)
    rgb_image = rgb_image.resize((32, 32))
    depth_image = depth_image.resize((32, 32))
    mask_image = mask_image.resize((32, 32))

    # Convert images to numpy arrays
    rgb_image = np.array(rgb_image)
    depth_image = np.array(depth_image)
    mask_image = np.array(mask_image)

    # Stack the images to create a 3-channel input
    stacked_image = np.stack((rgb_image, depth_image, mask_image), axis=-1)

    return stacked_image

# Load the CSV file
try:
    df = pd.read_csv(csv_file_path)
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Failed to load CSV file: {str(e)}")
    raise

# Map grip types to numeric labels
grip_type_mapping = {grip_type: idx for idx, grip_type in enumerate(df['grip_type'].unique())}
df['label'] = df['grip_type'].map(grip_type_mapping)

# Select 5 images from each unique object
test_df = df.groupby('grip_type').apply(lambda x: x.sample(n=1000, random_state=42)).reset_index(drop=True)

# Preprocess the selected objects
test_images = []
test_labels = []

for index, row in test_df.iterrows():
    image_base_name = extract_base_name(row['filename'])
    label = row['label']
    
    image = preprocess_image(image_base_name, target_folder)
    if image is not None:
        test_images.append(image)
        test_labels.append(int(label))
    else:
        print(f"Skipping image {image_base_name} due to missing file.")

test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_labels_categorical = tf.keras.utils.to_categorical(test_labels, num_classes=len(grip_type_mapping))

# Make predictions on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels_categorical, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Generate predictions and classification report
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
class_report = classification_report(test_labels, predicted_labels, target_names=grip_type_mapping.keys())
print("Classification Report:")
print(class_report)

# Save the classification report to a file
report_path = os.path.join(results_dir, "Classification report_for_paper 1000.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)
print(f"Classification report saved to {report_path}")

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=grip_type_mapping.keys(), yticklabels=grip_type_mapping.keys(),
            annot_kws={"size": 14})
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the confusion matrix as an image file
conf_matrix_path = os.path.join(results_dir, "Classification Matrix for Paper 1000.png")
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved to {conf_matrix_path}")

plt.show()

# Additional Metrics and Plots
# Calculate sensitivity (recall) and specificity for each class
def calculate_sensitivity_specificity(conf_matrix):
    sensitivities = []
    specificities = []

    for i in range(len(conf_matrix)):
        true_positive = conf_matrix[i, i]
        false_positive = conf_matrix[:, i].sum() - true_positive
        false_negative = conf_matrix[i, :].sum() - true_positive
        true_negative = conf_matrix.sum() - (true_positive + false_positive + false_negative)
        
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities

# Calculate sensitivities and specificities using the confusion matrix
sensitivities, specificities = calculate_sensitivity_specificity(conf_matrix)

# Print the sensitivity and specificity for each class
for i, class_name in enumerate(grip_type_mapping.keys()):
    print(f"Class: {class_name}, Sensitivity (Recall): {sensitivities[i]:.4f}, Specificity: {specificities[i]:.4f}")

# Plot ROC curve for each class
fpr = {}
tpr = {}
roc_auc = {}

# Binarize the labels for ROC computation
test_labels_binarized = label_binarize(test_labels, classes=list(grip_type_mapping.values()))

for i in range(len(grip_type_mapping)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
for i, color in zip(range(len(grip_type_mapping)), itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'purple', 'green', 'brown'])):
    # Plot ROC curve for each class
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {list(grip_type_mapping.keys())[i]} (area = {roc_auc[i]:.2f})')

# Plotting settings for ROC curves
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the ROC curve plot
roc_curve_path = os.path.join(results_dir, "roc_curve for paper  1000.png")
plt.savefig(roc_curve_path)
print(f"ROC curve plot saved to {roc_curve_path}")
plt.show()

# Plot Precision-Recall curves for each class
precision = {}
recall = {}
average_precision = {}

plt.figure(figsize=(10, 8))
for i in range(len(grip_type_mapping)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels_binarized[:, i], predictions[:, i])
    average_precision[i] = auc(recall[i], precision[i])
    plt.plot(recall[i], precision[i], lw=2, label=f'Precision-Recall curve of class {list(grip_type_mapping.keys())[i]} (area = {average_precision[i]:.2f})')

# Plotting settings for Precision-Recall curves
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curves', fontsize=16)
plt.legend(loc="lower left", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the Precision-Recall curve plot
pr_curve_path = os.path.join(results_dir, "precision_recall_curve for paper 1000.png")
plt.savefig(pr_curve_path)
print(f"Precision-Recall curve plot saved to {pr_curve_path}")
plt.show()

# Plot accuracy per class
accuracy_per_class = {}
for label, index in grip_type_mapping.items():
    class_accuracy = np.mean(predicted_labels[test_labels == index] == index)
    accuracy_per_class[label] = class_accuracy

plt.figure(figsize=(12, 8))
plt.bar(accuracy_per_class.keys(), accuracy_per_class.values())
plt.xlabel('Class', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy per Class', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Save the accuracy per class plot
accuracy_per_class_path = os.path.join(results_dir, "accuracy_per_class for paper 1000.png")
plt.savefig(accuracy_per_class_path)
print(f"Accuracy per class plot saved to {accuracy_per_class_path}")
plt.show()

# Plot Top-K accuracy (e.g., Top-2 accuracy)
top_k_accuracy_scores = []
for k in range(1, 3):  # Top-1 and Top-2
    top_k_predictions = np.argsort(predictions, axis=1)[:, -k:]
    top_k_accuracy = np.mean([test_labels[i] in top_k_predictions[i] for i in range(len(test_labels))])
    top_k_accuracy_scores.append(top_k_accuracy)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 3), top_k_accuracy_scores, marker='o')
plt.xlabel('Top-2', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Top-2 Accuracy', fontsize=16)
plt.xticks([1, 2], fontsize=12)
plt.yticks(fontsize=12)

# Save the Top-K accuracy plot
top_k_accuracy_path = os.path.join(results_dir, "top_k_accuracy  1000.png")
plt.savefig(top_k_accuracy_path)
print(f"Top-K accuracy plot saved to {top_k_accuracy_path}")
plt.show()

# t-SNE for Feature Visualization
features = model.predict(test_images, verbose=0)
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
for idx, label in enumerate(grip_type_mapping):
    plt.scatter(features_2d[test_labels == idx, 0], features_2d[test_labels == idx, 1], label=label)
plt.legend(fontsize=12)
plt.title('t-SNE Visualization of Model Features', fontsize=16)
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the t-SNE plot
tsne_path = os.path.join(results_dir, "tsne  1000.png")
plt.savefig(tsne_path)
print(f"t-SNE plot saved to {tsne_path}")
plt.show()

# Grad-CAM Visualization (for one example per class)
def plot_gradcam(image, model, layer_name):
    # Compute gradients
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[:, np.argmax(predictions)]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Compute CAM
    pooled_grads = np.mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], pooled_grads)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.imshow(image[..., 0], alpha=0.5)
    plt.title('Grad-CAM Visualization', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Generate Grad-CAM for one example per class
for class_index, class_name in grip_type_mapping.items():
    example_indices = np.where(test_labels == class_index)[0]
    if len(example_indices) > 0:
        example_index = example_indices[0]
        plot_gradcam(test_images[example_index], model, layer_name='conv2d')
    else:
        print(f"No examples found for class: {class_name}")
