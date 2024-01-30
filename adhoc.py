import torch
import torch.nn as nn
import torchvision.models as models

class SegmentationModel(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_encoder_layers=2):
        super(SegmentationModel, self).__init__()

        # ResNet18 feature extractor
        self.resnet18 = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(self.resnet18.children())[:-2])

        # Transformer-based encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        self.decoder = nn.ConvTranspose2d(d_model, num_classes, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)

        # Flatten features
        features = features.view(features.size(0), features.size(1), -1).permute(2, 0, 1)

        # Transformer-based encoding
        encoded_features = self.transformer_encoder(features)

        # Reshape for decoder
        encoded_features = encoded_features.permute(1, 2, 0).view(features.size(0), -1, features.size(2))

        # Decoder
        segmentation_mask = self.decoder(encoded_features)

        return segmentation_mask

# Instantiate the model
num_classes = 21  # Adjust based on your dataset
model = SegmentationModel(num_classes)

# Example usage
input_image = torch.randn(1, 3, 256, 256)  # Adjust the input size accordingly
output_mask = model(input_image)
