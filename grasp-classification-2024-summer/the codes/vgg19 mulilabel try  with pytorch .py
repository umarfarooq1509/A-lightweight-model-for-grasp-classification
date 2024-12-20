import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import traceback
from PIL import Image
from torch.utils.data import Dataset

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
batch_size = 32
epochs = 20
learning_rate = 0.0001
workers = 2  # Limit the number of workers to reduce CPU load

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset and split into training and testing sets (70-30 split)
df = pd.read_csv(csv_file_path)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# Custom Dataset
class MultilabelDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        labels = torch.tensor(self.dataframe.iloc[idx, 1:].values.astype('float32'))

        if self.transform:
            image = self.transform(image)

        return image, labels

# Data Augmentation and Normalization
train_transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(IMG_WIDTH, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = MultilabelDataset(dataframe=train_df, img_dir=train_dir, transform=train_transform)
val_dataset = MultilabelDataset(dataframe=val_df, img_dir=train_dir, transform=val_transform)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

# Load the VGG19 model without the top layers
base_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Modify the classifier to match the new task
class CustomVGG19(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomVGG19, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.features))
        self.avgpool = base_model.avgpool
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 512)  # 512 * 7 * 7 is the output size after avgpool
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = CustomVGG19(base_model, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and evaluation function
def train_and_evaluate():
    best_val_loss = float('inf')

    for epoch in range(epochs):
        try:
            # Training Phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy
                running_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).float().mean().item() * images.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            train_accuracy = correct / total

            # Validation Phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).float().mean().item() * images.size(0)
                    
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(predicted.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct / total

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(model_save_path, f'vgg19_model_epoch_{epoch+1}.pth'))

            # Log metrics to Discord
            message = (f"Epoch {epoch + 1}/{epochs}\n"
                       f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}\n"
                       f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
            requests.post(WEBHOOK_URL, json={'content': message})
            print(message)

            # Compute and save confusion matrix and classification report
            all_labels = np.vstack(all_labels)
            all_preds = np.vstack(all_preds)
            cm = confusion_matrix(all_labels.argmax(axis=1), all_preds.argmax(axis=1))
            cr = classification_report(all_labels, all_preds, target_names=train_df.columns[1:], output_dict=True)

            cm_save_path = os.path.join(model_save_path, f'confusion_matrix_epoch_{epoch + 1}.png')
            cr_save_path = os.path.join(model_save_path, f'classification_report_epoch_{epoch + 1}.json')

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(cm_save_path)
            plt.close()

            with open(cr_save_path, 'w') as f:
                pd.DataFrame(cr).transpose().to_json(f, indent=4)

        except Exception as e:
            # Send exception report to Discord
            exception_message = f"Exception occurred during epoch {epoch + 1}:\n{str(e)}\n{traceback.format_exc()}"
            requests.post(WEBHOOK_URL, json={'content': exception_message})
            print(exception_message)
            break

# Run training and evaluation
if __name__ == '__main__':
    try:
        train_and_evaluate()
    except Exception as e:
        exception_message = f"Exception occurred:\n{str(e)}\n{traceback.format_exc()}"
        requests.post(WEBHOOK_URL, json={'content': exception_message})
        print(exception_message)
