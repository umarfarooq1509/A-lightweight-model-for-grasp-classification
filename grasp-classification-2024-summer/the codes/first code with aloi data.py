import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress debugger warnings
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 10  # Additional epochs for further training
NUM_CLASSES = 4
DATA_DIR = "E:\\datasets"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations with data augmentation
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.class_names = sorted(os.listdir(directory))
        self.file_paths = []
        self.labels = []
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(directory, class_name)
            filenames = os.listdir(class_dir)
            for fname in filenames:
                self.file_paths.append(os.path.join(class_dir, fname))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

train_dataset = CustomDataset(DATA_DIR, transform=data_transforms)
train_size = int(0.7 * len(train_dataset))
val_size = int(0.2 * len(train_dataset))
test_size = len(train_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Define the two-layer CNN model
class TwoLayerCNN(nn.Module):
    def __init__(self):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)  # Updated to 3 channels for RGB input
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 53 * 53, 50)  # Updated input size
        self.fc2 = nn.Linear(50, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 53 * 53)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TwoLayerCNN().to(DEVICE)

# Load the previously saved model
model_path = 'epoch_6_model.pth'  # Replace with your model path
model.load_state_dict(torch.load(model_path))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Free up memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss, val_accuracy = evaluate_model(val_loader, model, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save the model after each epoch
        torch.save(model.state_dict(), f'epoch_{epoch + 1}_model.pth')

    return train_losses, val_losses, val_accuracies

# Evaluation function
def evaluate_model(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Free up memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
    
    loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    return loss, accuracy

if __name__ == '__main__':
    print("Starting training...")
    train_losses, val_losses, val_accuracies = train_model(model, criterion, optimizer, train_loader, val_loader, EPOCHS)
    print("Training complete.")

    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')

    # Evaluate on test set
    test_loss, test_accuracy = evaluate_model(test_loader, model, criterion, DEVICE)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Plot the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b', label='Training Loss')
    plt.plot(val_losses, 'r', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.legend()

    plt.show()
