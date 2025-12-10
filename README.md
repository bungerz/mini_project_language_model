# Mini Transformer Language Model

A small GPT-like model trained on Shakespeare data. Built from scratch to understand how transformers work.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   - Open `src/mini_transformer_model.ipynb` and Run All
   - Takes ~30 - 60 min on GPU (RTX 3070)

3. Generate text:
   - Use the generation cells at the end of the notebook
   - Try different temperatures to see how it affects output

## Structure

```
src/
- my_tests.py                   Unit tests
- my_tokenizer.py               Character-level tokenizer
- my_head.py                    Single attention head
- my_multihead.py               Multi-head attention
- my_ffn.py                     Feedforward network
- my_transformerblock.py        Transformer block
- my_gpt.py                     Main model class
- mini_transformer_model.ipynb  Training notebook

data/
- shakespeare.txt        # Training data

models/ (after running the model once)
- model.pth              # Best model weights
- checkpoint.pt          # Training checkpoint
```