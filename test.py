import torch
from torchvision import models

# Load EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)

# Remove the classifier to get only the feature extractor
feature_extractor = model.features

# Test with a sample input
x = torch.randn(1, 3, 256, 256)  # batch_size=1, channels=3, height=224, width=224
out = feature_extractor(x)

print(out.shape)  # Should output (1, 1280, 7, 7) for EfficientNet-B0
