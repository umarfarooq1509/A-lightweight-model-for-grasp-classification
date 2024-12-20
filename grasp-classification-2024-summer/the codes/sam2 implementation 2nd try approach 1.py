import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFile

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load SAM2 model (ViT-B in this example)
sam_checkpoint = "E:/codes/segment-anything/sam_vit_b.pth"  # Update with your path
model_type = "vit_b"  # Can be 'vit_b', 'vit_l', or 'vit_h'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Custom Dataset Class for Grasp Segmentation with Batch-wise Mask Generation
class GraspDatasetSegmented(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for i, cls in enumerate(classes):
            class_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):  # Check if it's a file
                    self.image_paths.append(img_path)
                    self.labels.append(i)
        
        # Log the number of samples
        logger.info(f"Loaded {len(self.image_paths)} images across {len(classes)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Error loading image at {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert image to PyTorch tensor and move it to the GPU
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).to(device)
            
            # Set the image for SAM2 and generate segmentation mask on GPU
            predictor.set_image(image_tensor)
            mask = predictor.predict()[0]

            # Convert mask to PyTorch tensor if it's not already
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(device)

            # Ensure the mask is 2D by taking the first channel if needed
            if mask.ndim == 3:
                mask = mask[0]  # Select the first channel if mask is 3D

            # Convert boolean mask to float for interpolation
            mask = mask.float()

            # Resize the mask to match the image dimensions
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=image_tensor.shape[1:], mode='nearest').squeeze()

            # Ensure the image is 3D (channels, height, width)
            if image_tensor.ndim == 2:
                image_tensor = torch.unsqueeze(image_tensor, 0)  # Add a channel dimension if missing

            # Apply mask to the image on GPU
            mask_expanded = mask.unsqueeze(0).repeat(3, 1, 1)
            masked_image = image_tensor * mask_expanded

            if self.transform:
                masked_image = self.transform(masked_image.cpu())

            return masked_image, label

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None, None

# Transformations for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def main():
    # Initialize Dataset and DataLoader
    root_dir = "E:/datasets/2022 paper/"
    classes = ["palmer pronated", "palmer neutral", "tripod", "key grasp", "index finger extension"]

    logger.info("Initializing dataset...")
    dataset = GraspDatasetSegmented(root_dir, classes, transform=transform)

    # Filter out None entries due to failed image processing
    dataset.image_paths, dataset.labels = zip(*[(img, lbl) for img, lbl in zip(dataset.image_paths, dataset.labels) if img is not None and lbl is not None])

    logger.info("Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)


    # Define the CNN Model for Grasp Classification
    class SegmentedGraspClassificationModel(nn.Module):
        def __init__(self):
            super(SegmentedGraspClassificationModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(128 * 16 * 16, 256)
            self.fc2 = nn.Linear(256, 5)  # 5 classes

        def forward(self, x):
            x = nn.ReLU()(self.conv1(x))
            x = nn.MaxPool2d(2)(x)
            x = nn.ReLU()(self.conv2(x))
            x = nn.MaxPool2d(2)(x)
            x = nn.ReLU()(self.conv3(x))
            x = nn.MaxPool2d(2)(x)
            x = x.view(x.size(0), -1)
            x = nn.ReLU()(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model, move it to the device, loss function, and optimizer
    model = SegmentedGraspClassificationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set up early stopping and model checkpointing
    checkpoint_dir = 'saved_models_of_sam2_approach1'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    best_val_loss = float('inf')

    # Training Loop with logging, checkpointing, and error handling
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for i, (images, labels) in enumerate(dataloader):
            if images is None or labels is None:
                continue  # Skip failed samples
            
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total

            # Log batch loss and accuracy
            logger.info(f"Batch {i + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")

        # Save the best model
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best model saved with loss {best_val_loss:.4f}")

    logger.info("Training completed.")

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info("Best model loaded.")

    # Evaluation on a Test Set
    def evaluate_model(model, dataloader):
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation for efficiency
            for images, labels in dataloader:
                if images is None or labels is None:
                    continue  # Skip failed samples

                # Move images and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total
        logger.info(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
        return avg_loss, accuracy

    # Prepare a test DataLoader (assuming you have a separate test set)
    test_dataset = GraspDatasetSegmented(root_dir, classes, transform=transform)  # Load your test dataset here
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    logger.info("Evaluating the model on the test set...")
    test_loss, test_accuracy = evaluate_model(model, test_dataloader)

    logger.info(f"Final Test Accuracy: {test_accuracy:.4f}, Final Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()
