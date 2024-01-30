import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, cnn_out_channels, transformer_d_model=256, transformer_nhead=8, transformer_layers=2):
        super(Encoder, self).__init__()

        # CNN-based Encoder
        self.cnn_block = CNNBlock(in_channels, cnn_out_channels)

        # Transformer-based Encoder
        self.positional_encoding = nn.Parameter(torch.randn(512, 512, transformer_d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=transformer_nhead),
            num_layers=transformer_layers
        )

    def forward(self, x):
        x = self.cnn_block(x)
        x = x.permute(2, 0, 1) + self.positional_encoding
        x = self.transformer_encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, transformer_d_model=256, num_classes=1):
        super(Decoder, self).__init__()

        # Decoder
        self.decoder = nn.ConvTranspose2d(transformer_d_model, num_classes, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.decoder(x)
        return F.sigmoid(x)  # Applying sigmoid for binary segmentation, adjust based on your task

class EncoderDecoderModel(nn.Module):
    def __init__(self, in_channels, cnn_out_channels, transformer_d_model=256, transformer_nhead=8, transformer_layers=2, num_classes=1):
        super(EncoderDecoderModel, self).__init__()

        # Encoder
        self.encoder = Encoder(in_channels, cnn_out_channels, transformer_d_model, transformer_nhead, transformer_layers)

        # Decoder
        self.decoder = Decoder(transformer_d_model, num_classes)

    def forward(self, x):
        encoded_features = self.encoder(x)
        segmentation_mask = self.decoder(encoded_features)
        return segmentation_mask

# Instantiate the model
in_channels = 3  # Adjust based on the number of input channels (e.g., 3 for RGB)
cnn_out_channels = 64  # Adjust based on your architecture
model = EncoderDecoderModel(in_channels, cnn_out_channels)

# Example usage
input_image = torch.randn(1, in_channels, 512, 512)  # Adjust the input size and channels accordingly
output_mask = model(input_image)
