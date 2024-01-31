import torch
import torch.nn as nn
import torchvision.models as models

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        # Ensure the input is treated as a sequence (sequence length is the height x width)
        b, c, h, w = src.shape
        src = src.view(b, c, -1).permute(2, 0, 1)  # Reshape to sequence length, batch, channels

        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        # Reshape back to original dimensions
        src = src.permute(1, 2, 0).view(b, c, h, w)

        return src

class SegmentationModelWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, transformer_layers=2, d_model=128, nhead=4):
        super(SegmentationModelWithTransformer, self).__init__()

        # Encoder with Transformer layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            TransformerEncoderLayer(d_model, nhead),
            nn.MaxPool2d(kernel_size=2, stride=2)
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
model = SegmentationModelWithTransformer(in_channels, out_channels)
output_tensor = model(input_tensor)

print("Input size:", input_tensor.shape)
print("Output size:", output_tensor.shape)

