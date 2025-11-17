import sys
from pathlib import Path

import pytest
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from my_tokenizer import CharDataset


def test_tokenizer_roundtrip(block_size=128):
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
    
    assert decoded == test_text, "Encoding/decoding broken!"
    assert dataset.get_vocab_size() > 0, "Vocab size should be positive"
    
    print("Test passed!")
