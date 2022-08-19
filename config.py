import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:

    # A general config class for all GPT models
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1

    def __init__(self, vocab_size, max_len, **kwargs):
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        for key, value in kwargs.items():
            setattr(self, key, value)

class GPT1Config(GPTConfig):
    num_heads = 12
    num_blocks = 12
    embed_dim = 768
    


