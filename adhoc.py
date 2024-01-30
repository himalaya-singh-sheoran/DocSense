import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTForImageSegmentation, ViTConfig
import torchvision.models as models

class SegmentationModel(nn.Module):
    def __init__(self, num_classes, vit_config):
        super(SegmentationModel, self).__init__()

        # ResNet18 feature extractor
        self.resnet18 = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(self.resnet18.children())[:-2])

        # Transformer-based encoder
        self.transformer_encoder = ViTForImageSegmentation(config=vit_config, num_labels=num_classes)

        # Decoder
        self.decoder = nn.ConvTranspose2d(256, num_classes, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)

        # Transformer-based encoding
        encoded_features = self.transformer_encoder(features)

        # Decoder
        segmentation_mask = self.decoder(encoded_features['last_hidden_state'])

        return segmentation_mask

# Instantiate the model
num_classes = 21  # Adjust based on your dataset
vit_config = ViTConfig()  # You may need to customize this based on your requirements
model = SegmentationModel(num_classes, vit_config)

# Example usage
input_image = torch.randn(1, 3, 256, 256)  # Adjust the input size accordingly
output_mask = model(input_image)
