import sys
from pathlib import Path

import pytest
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from my_head import Head
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
    print(f"Input text:  '{dataset.itos[x[0].item()]}{dataset.itos[x[1].item()]}{dataset.itos[x[2].item()]}{dataset.itos[x[3].item()]}'")
    print(f"Target text: '{dataset.itos[y[0].item()]}{dataset.itos[y[1].item()]}{dataset.itos[y[2].item()]}{dataset.itos[y[3].item()]}'")
    
    print("Test passed!")

def test_single_attention_head():
    # TEST: Single Attention Head
    print("Testing single attention Head")

    # Create dummy input
    batch_size, seq_len, embed_dim = 4, 8, 64
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create attention head
    head = Head(head_size=16, n_embd=embed_dim, block_size=128, dropout=0.0)
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
