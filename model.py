import torch 
import torch.nn as nn
from transformers.activations import gelu_new

class Gelu(nn.Module):

    def forward(self, X):
        return gelu_new(X)

class DecoderBlock(nn.Module):

    def __init__(
        self, 
        *, 
        emb_dim, 
        n_heads, 
        max_tokens, 
        attn_drop, 
        ffn_drop, 
        layer_norm_epsilon
        ):

        super().__init__()

        self.layer_norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
                                embed_dim = emb_dim, 
                                num_heads = n_heads,
                                dropout=attn_drop,
                                bias=True,
                                batch_first=True
                            )
        self.mask_inp(
            "mask",
            (1 - torch.tril(torch.ones(max_tokens, max_tokens))).to(
                dtype=torch.bool
            )
        )
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            Gelu(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(ffn_drop)
        )

    def forward(self, X):
        
        batch_size, n_tokens, emb_dim = X.shape
        X_ln1 = self.layer_norm1(X)
        mask = self.mask[:n_tokens, :n_tokens]
        attn_out = self.attn(
            X_ln1, X_ln1, X_ln1, attn_mask=mask, need_weights=False
        )
        X = X + attn_out
        X = X + self.ffn(self.layer_norm2(X))

        return X