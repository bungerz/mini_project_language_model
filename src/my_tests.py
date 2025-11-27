import sys
from pathlib import Path

import pytest
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from my_head import Head
from my_multihead import MultiHead
from my_tokenizer import CharDataset


def test_tokenizer_roundtrip(block_size=4):
    print("Testing tokenizer encoding / decoding ")
    test_text = "Hello, world!"
    
    # Create dataset
    dataset = CharDataset(test_text, block_size=block_size)
    
    # Test encode/decode
    encoded = [dataset.stoi[c] for c in test_text]
    decoded = dataset.decode(encoded)
    
    print(f"Original:  {test_text}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded}")
    print(f"Vocab size: {dataset.get_vocab_size()}")
    print(f"Dataset length: {len(dataset)}")
    
    assert decoded == test_text, "Encoding/decoding broken"
    #assert dataset.get_vocab_size() > 0, "Vocab size should be positive"
    
    # Get first sample
    x, y = dataset[0]
    print(f"\nFirst sample:")
    print(f"Input:  {x}")
    print(f"Target: {y}")

    # Decode to verify
    input_text = ''.join([dataset.itos[idx.item()] for idx in x])
    target_text = ''.join([dataset.itos[idx.item()] for idx in y])
    print(f"Input text:  '{input_text}'")
    print(f"Target text: '{target_text}'")
    
    print("Test passed!")

def test_single_attention_head():
    head_size = 16
    print(f"Testing single attention Head with head size {head_size}")

    # Create dummy input
    batch_size, seq_len, embed_dim = 4, 8, 64
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create attention head
    head = Head(head_size=head_size, n_embd=embed_dim, block_size=128, dropout=0.0)
    output = head(x)

    # Test 1: Shape
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 16), "Output head_size should be 16"

    # Test 2: No NaN values
    assert not torch.isnan(output).any(), "NaN values in output"

    # Test 3: Output changes with different input
    x2 = torch.randn(batch_size, seq_len, embed_dim)
    output2 = head(x2)
    assert not torch.allclose(output, output2), "Output doesn't depend on input"

    # Test 4: Causal masking works (manual check)
    # Position 0 should only attend to itself
    # Position 5 should only attend to positions 0-5
    print("Single attention head works")

def test_multi_attention_head():
    head_size = 16
    block_size = 128
    dropout = 0.0
    num_heads = 4
    # TEST: Multi-Head Attention
    print(f"Testing Multi-Head Attention with head size {head_size} and num heads {num_heads}")

    # Create dummy input
    batch_size, seq_len, embed_dim = 4, 8, 64
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create multi-head attention
    multi_head = MultiHead(num_heads, head_size=head_size, n_embd=embed_dim, 
                                    block_size=block_size, dropout=dropout)
    output = multi_head(x)

    # Test 1: Shape
    assert output.shape == x.shape, "Shape mismatch"

    # Test 2: No NaN
    assert not torch.isnan(output).any(), "NaN values"

    print("Multi-head attention works")