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

class SegmentationModelWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, transformer_layers=2, d_model=128, nhead=4):
        super(SegmentationModelWithTransformer, self).__init__()

        # Encoder with Transformer layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead) for _ in range(transformer_layers)
        ])

        # Decoder remains the same
        # ...

    def forward(self, x):
        x = self.encoder(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

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

