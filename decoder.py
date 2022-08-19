import torch 
import torch.nn as nn
from attention import MultiHeadAttention

class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.attention = MultiHeadAttention(config)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 4*self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim*4, self.embed_dim),
            nn.Dropout(config.ff_dropout)
        )
    
    def forward(self, x):

        x = x + self.attention(self.ln1(x))
        x = x + self.head(self.ln2(x))

        return x