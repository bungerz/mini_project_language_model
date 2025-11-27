import sys
from pathlib import Path

import pytest
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from my_ffn import FeedForward
from my_head import Head
from my_multihead import MultiHead
from my_tokenizer import CharDataset
from my_transformerblock import TransformerBlock


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
    
    x_future_modified = x.clone()
    x_future_modified[:, 4:, :] = torch.randn(batch_size, seq_len - 4, embed_dim)
    output_future_modified = head(x_future_modified)
    
    assert torch.allclose(output[:, 0, :], output_future_modified[:, 0, :], atol=1e-5), "Causal masking broken: position 0 affected by future"
    assert torch.allclose(output[:, 3, :], output_future_modified[:, 3, :], atol=1e-5), "Causal masking broken: position 3 affected by future"
    print("Causal masking verified")
    
    print("Single attention head works\n")

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
    print(f"Shape preserved: {x.shape} to {output.shape}")
    
    # Test 2: No NaN
    assert not torch.isnan(output).any(), "NaN values"
    print("No NaN values")

    # Test 3: Output depends on input
    x2 = torch.randn(batch_size, seq_len, embed_dim)
    output2 = multi_head(x2)
    assert not torch.allclose(output, output2), "Output doesn't depend on input"
    print("Output changes with different input")
    
    # Test 4: Projection layer dimensions
    assert multi_head.proj.in_features == head_size * num_heads, "Projection input wrong"
    assert multi_head.proj.out_features == embed_dim, "Projection output wrong"
    print(f"Projection: {head_size * num_heads} to {embed_dim}")
    
    # Test 5: Causal masking (future doesn't affect past)
    x_future_modified = x.clone()
    x_future_modified[:, 4:, :] = torch.randn(batch_size, seq_len - 4, embed_dim)
    output_future_modified = multi_head(x_future_modified)
    assert torch.allclose(output[:, 0, :], output_future_modified[:, 0, :], atol=1e-5), "Causal masking broken: position 0 affected by future"
    print("Causal masking verified")

    print("Multi-head attention works!\n")
    
def test_ffn():
    print("Testing Feed-Forward Network...")

    # Create dummy input
    batch_size, seq_len, embed_dim = 4, 8, 32
    x = torch.randn(batch_size, seq_len, embed_dim)
    ffn = FeedForward(n_embd=embed_dim, dropout=0.0)
    output = ffn(x)

    # Test 1: Shape preservation
    assert output.shape == x.shape, "Shape mismatch"
    print(f"Shape preserved: {x.shape} to {output.shape}")

    # Test 2: No NaN
    assert not torch.isnan(output).any(), "NaN values"
    print("No NaN values")

    # Test 3: Output depends on input
    x2 = torch.randn(batch_size, seq_len, embed_dim)
    output2 = ffn(x2)
    assert not torch.allclose(output, output2), "Output doesn't depend on input"
    print("Output changes with different input")

    # Test 4: Verify 4x expansion structure
    assert ffn.ffn[0].out_features == 4 * embed_dim, "Should expand to 4x"
    assert ffn.ffn[2].out_features == embed_dim, "Should compress back"
    print(f"Internal structure: {embed_dim} to {4*embed_dim} to {embed_dim}")

    # Test 5: Position-wise independence
    x_modified = x.clone()
    x_modified[:, 0, :] = torch.randn(batch_size, embed_dim)
    output_modified = ffn(x_modified)
    assert torch.allclose(output[:, 1:, :], output_modified[:, 1:, :]), "FFN should be position-wise independent"
    print("Position-wise independence verified")

    print("Feed-forward network works!\n")
    
def test_transformer_block():
    print("Testing Transformer Block...")

    batch_size, seq_len, embed_dim = 4, 8, 32
    x = torch.randn(batch_size, seq_len, embed_dim)
    block = TransformerBlock(n_embd=32, num_heads=4, block_size=128, dropout=0.0)
    output = block(x)

    # Test 1: Shape preservation
    assert output.shape == x.shape, "Shape mismatch"
    print(f"Shape preserved: {x.shape} to {output.shape}")
    
    # Test 2: No NaN
    assert not torch.isnan(output).any(), "NaN values"
    print("No NaN values")

    # Test 3: Stacking blocks
    block2 = TransformerBlock(n_embd=32, num_heads=4, block_size=128, dropout=0.0)
    output2 = block2(output)
    assert output2.shape == x.shape, "Can't stack blocks!"

    print("Transformer block works")