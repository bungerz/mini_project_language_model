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
            x: Input tensor of shape (Batch_size, Time (block_size), Channels (n_embd:embedding dimension))
        Returns:
            Output tensor of shape (Batch_size, Time (block_size), head_size)
        """
        
        B, T, C = x.shape
        
        # 1. Compute keys, queries, values
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # 2.0 Compute attention scores QK^T / sqrt(d_k)
        # q: (B, T, head_size), k.permute: (B, head_size, T) gives att: (B, T, T)
        att = (q @ k.permute(0, 2, 1)) / (sqrt(self.head_size))
        
        # 2.1 causal Mask, we dont look at future tokens: softmax(-inf) = 0
        att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        
        #2.2 apply softmax to turn result unto probabilities
        # # dim=-1 normalizes across the key dimension, so each query position's attention weights sum to 1  (dim-2 being the query dimension)
        att = nn.functional.softmax(att, dim=-1)
        
        # 2.3 apply dropout
        att = self.dropout(att)
        
        # 2.4  attention weights to values: (B, T, T) @ (B, T, head_size) gives (B, T, head_size)
        return att @ v