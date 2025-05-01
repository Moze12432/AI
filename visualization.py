import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
# Set device to GPU if available
device = torch.device("cpu")

# 1. Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (input size for ResNet)
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# 2. Load the Dataset
train_data = datasets.ImageFolder(root='AI/data/train', transform=transform)
val_data = datasets.ImageFolder(root='AI/data/val', transform=transform)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 3. Load Pretrained ResNet Model
# Use the new weights parameter instead of the deprecated pretrained parameter
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device)

# Modify the final fully connected layer to match the number of classes (e.g., 2 for "healthy" and "defective")
num_classes = 2  # Update this based on your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes, device=device)

# Move model to device


# 4. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

writer = SummaryWriter('runs/gender_detector_experiment_1')
dataiter = iter(train_loader)
images, labels = next(dataiter)

writer.add_graph(model, images)

writer.flush()
writer.close()

# 5. Training the Model
# num_epochs = 10
# for epoch in range(num_epochs):
    # model.train()  # Set model to training mode
    # running_loss = 0.0
    
    # for images, labels in train_loader:
        # images, labels = images.to(device), labels.to(device)  # Move data to device
        
        # optimizer.zero_grad()  # Clear previous gradients
        # outputs = model(images)  # Forward pass
        # loss = criterion(outputs, labels)  # Calculate loss
        # loss.backward()  # Backward pass
        # optimizer.step()  # Update weights
        
        # running_loss += loss.item()
    
    # # Print average loss per epoch
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# # 6. Evaluating the Model
# model.eval()  # Set model to evaluation mode
# all_preds = []
# all_labels = []

# with torch.no_grad():
    # for images, labels in val_loader:
        # images, labels = images.to(device), labels.to(device)
        # outputs = model(images)
        # _, preds = torch.max(outputs, 1)
        # all_preds.extend(preds.cpu().numpy())
        # all_labels.extend(labels.cpu().numpy())

# # Calculate accuracy
# accuracy = accuracy_score(all_labels, all_preds)
# print(f"Validation Accuracy: {accuracy:.4f}")
# # Save the model
# model_save_path = os.path.join("AI", "model", "gender_detector.pth")
# torch.save(model.state_dict(), model_save_path)
# print(f'Model saved at {model_save_path}')

