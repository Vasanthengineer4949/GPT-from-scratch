import torch
from gpt import GPT
from config import GPT1Config

if __name__ == "__main__":
    vocab_size = 10
    max_len = 12
    config = GPT1Config(vocab_size, max_len)
    model = GPT(config)
    seq_len = 15
    batch_size = 3
    num_heads = 2

    test_input = torch.randint(high=vocab_size, size=(batch_size, seq_len))
    try:
        print(test_input)
        print(model(test_input[:, :max_len]).shape)
    except AssertionError as e:
        print(e)

