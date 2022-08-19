import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim # Dimension of embedding
        self.num_heads = config.num_heads # Number of attention heads

        self.query = nn.Linear(self.embed_dim, self.embed_dim) # Query vector
        self.key = nn.Linear(self.embed_dim, self.embed_dim) # Key vector
        self.value = nn.Linear(self.embed_dim, self.embed_dim) # Value vector

        self.ff = nn.Linear(self.embed_dim, self.embed_dim) # Output projection from FFN

        self.attn_dropout = nn.Dropout(config.attn_dropout) # Attention Dropout
        self.ff_dropout = nn.Dropout(config.ff_dropout) # Feed forward NN dropout

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len))
            .unsqueeze(0).unsqueeze(0)
        ) # Mask buffer for masking attention

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Reshaping query, key and value vectors for different attention heads
        query = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        key = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = self.value.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Scaled Dot Product Attention but adapted masking
        attn = torch.matmul((query, key.transpose(-2, -1))/math.sqrt(key.size(-1))) #Q.Kt/root(dmodel)
        mask = self.mask[:, :, :seq_len, :seq_len] # mask
        attn_masked = attn.masked_fill(mask==0, float("-inf")) # fill attention with mask
        attn_dropped = self.attn_dropout(attn_masked) # perform dropout on attention
        attn_weights = F.softmax(attn_dropped, dim=-1) # softmax(attention) to get attention weights
        attn_score = torch.matmul(attn_weights, value) # attention score = softmax(attention_score)*value

        # Pass the attention to a FFN where all the attention from different heads are concatenated
        concat_attn_inp = attn_score.transpose(1, 2)
        concat_attn_inp_reshaped = concat_attn_inp.reshape(batch_size, seq_len, -1)
        attention_out = self.ff_dropout(concat_attn_inp_reshaped)

        return attention_out

        




        