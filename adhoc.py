import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers
        )

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_heads),
            num_layers
        )

    def forward(self, memory, tgt):
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.transformer_decoder(tgt, memory)
        return output

class SegmentationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(SegmentationModel, self).__init__()
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads)
        self.decoder = TransformerDecoder(output_dim, hidden_dim, num_layers, num_heads)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(memory, tgt)
        return output

# Example usage:
input_dim = 1000  # Input vocabulary size
output_dim = 2    # Output vocabulary size (assuming binary segmentation mask)
hidden_dim = 256  # Hidden dimension
num_layers = 4    # Number of layers
num_heads = 8     # Number of attention heads

model = SegmentationModel(input_dim, output_dim, hidden_dim, num_layers, num_heads)

