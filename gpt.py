from lib2to3.pgen2 import token
from model import DecoderBlock
import torch
import torch.nn as nn

class GPT(nn.Module):

    def __init__(
        self,
        *,
        vocab_size,
        n_layers,
        emb_dim,
        n_heads,
        pos_dim,
        attn_drop,
        embd_drop,
        ffn_drop,
        layer_norm_epsilon
    ):
        super(GPT, self).__init__()
        self.pos_dim = pos_dim
        self.token_emb  = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(pos_dim, emb_dim)
        self.dropout  = nn.Dropout(embd_drop)
        self.decoder = nn.Sequential(
            *[
                DecoderBlock(
                    emb_dim,
                    n_heads,
                    pos_dim,
                    attn_drop,
                    ffn_drop,
                    layer_norm_epsilon
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(emb_dim, layer_norm_epsilon)
        self.ffn = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, X):
        
        batch_size, n_tokens = X.shape
        device=  X.device

        if n_tokens > self.pos_dim:
            raise ValueError("Input token cant be too long")

        positions = torch.arange(n_tokens, device=device)
        token_emb = self.token_emb(X)
        pos_emb = self.pos_emb(positions)[None, ...]
        embed = token_emb + pos_emb
        embed_inp = self.dropout(embed)
        dec_out = self.decoder(embed_inp)
        dec_out_norm = self.layer_norm(dec_out)
        logits = self.ffn(dec_out_norm)

        return logits


