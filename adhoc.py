import torch
import torch.nn as nn
from torchvision import models

class UNetWithViT(nn.Module):
    def __init__(self, in_channels, out_channels, vit_model='vit_base_patch16_224_in21k'):
        super(UNetWithViT, self).__init__()

        # Use a Vision Transformer as the encoder
        self.encoder = models.vit.__dict__[vit_model](pretrained=True, num_classes=0)  # num_classes=0 removes the classification head

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # Use only the features from the vision transformer as input to the decoder
        x = self.encoder(x)['last_hidden_state']
        x = self.decoder(x)
        return x

# Instantiate the model
in_channels = 3  # Input channels (e.g., for RGB images)
out_channels = 1  # Output channels for binary segmentation

model = UNetWithViT(in_channels, out_channels)

