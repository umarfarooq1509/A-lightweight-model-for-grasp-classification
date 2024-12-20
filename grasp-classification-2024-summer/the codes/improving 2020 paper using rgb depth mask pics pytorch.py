import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import time
from threading import Timer
import cv2
from torchvision import transforms
from torchvision.utils import save_image

# Ensure that GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your target folder and the CSV file
target_folder = r"E:\datasets\rgbd-dataset 2.0\multilabel_dataset"
csv_file_path = r"E:\datasets\rgbd-dataset 2.0\grip_labels_2.0.csv"
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

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, target_folder, transform=None):
        self.dataframe = dataframe
        self.target_folder = target_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_base_name = "_".join(self.dataframe.iloc[idx]['filename'].split('_')[:-1])
        label = int(self.dataframe.iloc[idx]['label'])
        
        # Preprocess images
        rgb_image_path = os.path.join(self.target_folder, f"{image_base_name}_crop.png")
        depth_image_path = os.path.join(self.target_folder, f"{image_base_name}_depthcrop.png")
        mask_image_path = os.path.join(self.target_folder, f"{image_base_name}_maskcrop.png")
        
        rgb_image = cv2.imread(rgb_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

        if rgb_image is None or depth_image is None or mask_image is None:
            raise ValueError(f"One or more images could not be loaded: {rgb_image_path}, {depth_image_path}, {mask_image_path}")

        depth_image = cv2.medianBlur(depth_image, 5)

        rgb_image = cv2.resize(rgb_image, (32, 32))
        depth_image = cv2.resize(depth_image, (32, 32))
        mask_image = cv2.resize(mask_image, (32, 32))

        rgb_image_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        stacked_image = np.stack((rgb_image_gray, depth_image, mask_image), axis=-1)

        if self.transform:
            stacked_image = self.transform(stacked_image)

        return stacked_image, label

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

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create Datasets and DataLoaders
train_dataset = CustomDataset(train_df, target_folder, transform=transform)
val_dataset = CustomDataset(val_df, target_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleCNN(num_classes=len(grip_type_mapping)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        progress_message = (
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )
        print(progress_message)
        send_to_discord(progress_message)

        # Save the model after each epoch
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)

        # Generate confusion matrix and classification report
        generate_metrics(model, val_loader, epoch)

# Generate confusion matrix and classification report
def generate_metrics(model, val_loader, epoch):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(grip_type_mapping.keys()), output_dict=True)

    # Save confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(grip_type_mapping.keys()), yticklabels=list(grip_type_mapping.keys()))
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
    plt.close()

    # Save classification report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, f'classification_report_epoch_{epoch+1}.csv'))

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
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs)
except Exception as e:
    error_message = f"Training failed: {str(e)}"
    print(error_message)
    send_to_discord(error_message)
    raise

# Final evaluation after training is completed
model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss = val_loss / len(val_loader)
val_accuracy = correct / total
final_message = f"Final validation accuracy: {val_accuracy:.4f}, Validation loss: {val_loss:.4f}"
print(final_message)
send_to_discord(final_message)

# Final saving of the trained model
final_model_save_path = os.path.join(save_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_save_path)
final_save_message = f"Final model saved to {final_model_save_path}"
print(final_save_message)
send_to_discord(final_save_message)

