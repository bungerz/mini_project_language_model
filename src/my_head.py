from math import sqrt

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        """
        Args:
            head_size: Dimensions of Q,K,V projections
            n_embd: the embedding dimension
            block_size: max sequence lenght
            dropout: dropout rate
        """
        super().__init__()
        self.head_size = head_size
        
        ## 1. Register the layers
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        
        #2. Create the lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        #3. Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, time (seq length), channels (embedding dimension))
        Returns:
            Output tensor of shape (batch_size, time (seq length), head_size)
        """
        
        B, T, C = x.shape
        
        # 1. Compute keys, queries, values
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # 2. Compute attention scores (Attention(Q, K, V) = softmax(QK / sqrt(d_k)) V)
        # 2.0 inside parenthesis
        # both q and k are initially  (B, T, head_size), we transpose k to be in the shape of (B, head_size, T)
        att = (q @ k.permute(0, 2, 1)) / (sqrt(self.head_size)) # new shape : (B, T (seq len), T (seq len))
        
        # 2.1 causal Mask
        att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        
        #2.2 apply softmax to turn result unto probabilities
        att = nn.functional.softmax(att, dim=-1)
        
        # 2.3 apply dropout
        att = self.dropout(att)
        
        # 2.4 multiply by values        
        return att @ v