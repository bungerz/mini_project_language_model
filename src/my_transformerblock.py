from numpy import block
from torch import nn

from my_ffn import FeedForward
from my_multihead import MultiHead


class TransformerBlock(nn.Module):
    """Transformer block: attention + feed-forward with residuals"""
    def __init__(self, num_heads, n_embd, block_size, dropout):
        """
        Args:
            num_heads: Number of attention heads
            n_embd: Embedding dimension
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        #0. Retrieving the head_size automatically
        head_size = n_embd // num_heads
        
        #1. Multi-head attention
        self.att = MultiHead(num_heads, head_size, n_embd, block_size, dropout)
        
        #2. FFN
        self.ffn = FeedForward(n_embd,dropout)
        
        #3. Layer Norm : normalize across features
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch, time, n_embd)
        Returns:
            Output of shape (batch, time, n_embd)
        """
        #1. We apply the attention pre-norm and residual (as nanoGPT)
        x = x + self.att(self.ln1(x)) # 'Attention is all your need' is post-norm : x = self.ln1(x + self.att(x)) 
        #2. Same for FFN
        x = x + self.ffn(self.ln2(x))
        
        return x
        