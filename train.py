import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress bar
from dataloader import SegmentationDataset
from model import SegmentationModel
from torchvision import transforms
from torch.utils.data import DataLoader

# Define the dataset
image_dir = '/Users/hiteshgupta/Documents/ML-CV/Image-Segement-Forgery/Dataset/val/img'
mask_dir = '/Users/hiteshgupta/Documents/ML-CV/Image-Segement-Forgery/Dataset/val/mask'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Create DataLoader
batch_size = 8  # You can adjust this based on your system memory
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
model = SegmentationModel(input_size=(256, 256))  # Adjust input size if necessary
model = model.to(device)  # Move the model to MPS or CPU

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Set the number of epochs

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)  # Move images to MPS/CPU
        masks = masks.to(device)  # Move masks to MPS/CPU
        
        # Forward pass
        outputs = model(images)
        
        # Reshape outputs and masks to have the same shape (batch_size, num_pixels)
        outputs = outputs.view(-1)
        masks = masks.view(-1)
        
        # Compute loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
