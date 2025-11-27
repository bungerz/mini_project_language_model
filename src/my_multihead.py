import torch
from torch import nn

import my_head
from my_head import Head


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        """
        Args:
            num_heads: Number of attention heads
            head_size: Dimensions of Q,K,V projections
            n_embd: the embedding dimension
            block_size: max sequence lenght
            dropout: dropout rate
        """
        super().__init__()
        # 1. Create the list of attention heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        
        # 2. Create projection layer for after concatenation
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        
        # 3. Create dropout
        self.dropout = nn.Dropout(dropout)
            
    def forward(self,x: torch.Tensor):
        """
        Args:
            x: Input of shape (batch, time, n_embd)
        Returns:
            Output of shape (batch, time, n_embd)
        """        
        #1. Run heads in parallel and concatenated
        # We concatenate along the feature dimension
        out = torch.concat([head(x) for head in self.heads], dim=-1)
        
        #2. Apply dropout
        out = self.dropout(out)
        
        #3. Project back to n_embd dimension
        out = self.proj(out)
        
        return out
        
        

        
    
        
        
    