import sys
from pathlib import Path

import pytest
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from my_tokenizer import CharDataset


def test_tokenizer_roundtrip(block_size=4):
    """Test that encode/decode works correctly"""
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
    
    assert decoded == test_text, "Encoding/decoding broken!"
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
