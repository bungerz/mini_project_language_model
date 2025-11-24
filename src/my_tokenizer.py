import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """    
    def __init__(self, data: str, block_size):

        chars = sorted(list(set([a_char for a_char in data]))) # get characters from the input data
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices
        self.itos = { i:ch for i,ch in enumerate(chars) } # map integer indices to characters
        self.block_size = block_size
        self.data = data

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        encoded_chunk = [self.stoi[c] for c in chunk]
        # return the chunk and the shifted version as tensors
        return torch.tensor(encoded_chunk[:-1]), torch.tensor(encoded_chunk[1:])

    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices])