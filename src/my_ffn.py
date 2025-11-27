from torch import nn


# 3.3 Position-wise Feed-Forward Networks
class FeedForward(nn.Module):
    # "This consists of two linear transformations with a ReLU activation in between".
    """Position-wise feed-forward network"""
    def __init__(self, n_embd, dropout):
        """
        Args:
            n_embd: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        # 1. Create the FFN "The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df f = 2048"
        self.ffn = nn.Sequential(nn.Linear(n_embd,4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    
        # 2. Create dropout
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch, time, n_embd)
        Returns:
            Output of shape (batch, time, n_embd)
        """  
        out = self.ffn(x)
        out = self.dropout(out)
        return out
        
        
        