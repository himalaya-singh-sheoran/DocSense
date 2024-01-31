import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.feedforward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class DenoisingModel(nn.Module):
    def __init__(self, in_channels, out_channels, transformer_layers=2, d_model=128, nhead=4):
        super(DenoisingModel, self).__init__()

        # Transformer Encoder with additional convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            TransformerEncoderLayer(d_model, nhead),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Assuming the output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
in_channels = 3  # Input channels (e.g., for RGB images)
out_channels = 3  # Output channels for denoising (RGB)

# Ensure the model can handle 256x256 images (adjust as needed)
input_tensor = torch.randn((1, in_channels, 256, 256))
model = DenoisingModel(in_channels, out_channels)
output_tensor = model(input_tensor)

print("Input size:", input_tensor.shape)
print("Output size:", output_tensor.shape)
