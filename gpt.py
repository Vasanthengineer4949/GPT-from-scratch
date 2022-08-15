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
        self.token_emb  = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(pos_dim, emb_dim)
        self.dropout  = nn.Dropout(embd_drop)
        self.blocks = nn.Sequential(
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
        
