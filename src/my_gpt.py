import torch
from torch import nn

import my_tokenizer
from my_tokenizer import CharDataset
from my_transformerblock import TransformerBlock


class SmolGPT(nn.Module):
    "Full transformer model"
    def __init__(self, vocab_size, n_embd, block_size, num_head, num_layers, dropout):
        """
        Args:
            vocab_size: Size of vocabulary
            n_embd: Embedding dimension
            block_size: Maximum sequence length
            num_head: Number of attention heads per block
            num_layers: Number of transformer blocks
            dropout: Dropout probability
        """
        super().__init__()
        # For cropping sequences in generate()
        self.block_size = block_size 
        #1. Token Embeddings (from vocab_size to n_embd) :
        #"we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model"
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        #2. Position Embeddings (from block_size to n_embd)
        #"We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results"
        # TODO: Replace with "sine and cosine functions of different frequencies"
        self.position_embeddings = nn.Embedding(block_size, n_embd)
        
        #4. N layers of Transformers blocks
        ## nn.Sequential takes a list of arguments so we unpack a list comprehension
        self.blocks = nn.Sequential(
            * [TransformerBlock(num_head,n_embd,block_size,dropout) for _ in range(num_layers)]
        )
        #5. Layer Norm
        self.ln = nn.LayerNorm(n_embd)
        #6. From embedding back to vocabulary (n_embd to vocab_size)
        self.head = nn.Linear(n_embd, vocab_size)
        
    # for training
    def forward(self, idx, targets=None):
        """
        Args:
            idx: Input token indices of shape (batch, time)
            targets: Target token indices of shape (batch, time) [optional]
        Returns:
            logits: Predictions of shape (batch, time, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = idx.shape
        
        #1. Get token embeddings
        tok_emb = self.token_embeddings(idx)
        #2. Get position embeddings
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embeddings(positions)
        #3. Add both
        x = tok_emb + pos_emb  #Auto broadcast (B, T, n_embd) + (T, n_embd)
        #4. Transformers blocks + Layer Norm
        x = self.ln(self.blocks(x))
        #5. Project to vocabulary
        logits = self.head(x)
        
        #6. We compute the loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    
        return logits, loss
    
    # for inference
    def generate(self,idx, n_new_tokens):
        #TODO: add temperature
        """
        Generate new tokens autoregressively
        
        Args:
            idx: Starting sequence of shape (batch, time)
            n_new_tokens: Number of tokens to generate
            temperature: sampling temperature (higher = more random)
        Returns:
            Generated sequence of shape (batch, time + n_new_tokens)"""
        for _ in range(n_new_tokens):
            #1. get predictions for all positions (batch, seq_len, vocab_size)
            #1.1 crop to block_size if sequence too long
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            
            #2. we only want the last logits seq_len wise (batch, vocab_size)
            logits = logits[:,-1,:]
            
            #3. we SAMPLE the next token
            proba = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(proba, num_samples=1)
            
            # Append to the sequence to get the next one
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx
            