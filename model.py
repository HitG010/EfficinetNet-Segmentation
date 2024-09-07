import torch
import torch.nn as nn
from torchvision import models

class SegmentationModel(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(SegmentationModel, self).__init__()
        
        # Load a pretrained EfficientNet (e.g., EfficientNet-B0)
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.encoder = efficientnet.features  # Use EfficientNet feature extractor
        
        # Get feature map output size from EfficientNet
        self.encoder_output_channels = 1280  # For EfficientNet-B0, the output has 1280 channels
        
        # Decoder - upsample the encoder's output back to the input size
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder_output_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output 1 channel for binary mask
            nn.Sigmoid()  # Sigmoid for binary segmentation
        )

    def forward(self, x):
        # Encoder: EfficientNet feature extraction
        x = self.encoder(x)
        # print(x.shape)
        
        # Decoder: Upsample back to input size
        x = self.decoder(x)
        return x

# Instantiate the model for binary segmentation
model = SegmentationModel(input_size=(224, 224))

# Test with a dummy input
input_image = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3, height=224, width=224
output = model(input_image)

print(f"Output shape: {output.shape}")  # Should output (1, 1, 224, 224)
