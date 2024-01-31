import torch
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Using Sigmoid activation for binary segmentation
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
in_channels = 3  # Input channels (e.g., for RGB images)
out_channels = 1  # Output channels for binary segmentation

# Ensure the model can handle 512x512 images
input_tensor = torch.randn((1, in_channels, 512, 512))
model = SegmentationModel(in_channels, out_channels)
output_tensor = model(input_tensor)

print("Input size:", input_tensor.shape)
print("Output size:", output_tensor.shape)

