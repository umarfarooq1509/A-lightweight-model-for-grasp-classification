import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress debugger warnings
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32  # Reduced batch size for better memory management
EPOCHS = 20
NUM_CLASSES = 3
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Model Building with VGG16 for transfer learning
class VGG16Transfer(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Transfer, self).__init__()
        self.base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.classifier[6] = nn.Linear(self.base_model.classifier[6].in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

model = VGG16Transfer(NUM_CLASSES).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.base_model.classifier.parameters(), lr=0.0001)

# Training function
def train_model(model, criterion, optimizer, dataloader, num_epochs):
    model.train()
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}/{num_epochs}")
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Free up memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs }, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        
        # Save the model after every epoch
        torch.save(model.state_dict(), f'epoch_{epoch}_pytorch_model.pth')
        
        # Store the metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
    
    return train_losses, train_accuracies

# Main block
if __name__ == '__main__':
    print("Starting training... 0")
    train_losses, train_accuracies = train_model(model, criterion, optimizer, train_loader, EPOCHS)
    print("Training complete.")
    # Save the final model
    torch.save(model.state_dict(), 'final_pytorch_model.pth')
    
    # Plot the results after training is complete
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'r', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.show()
