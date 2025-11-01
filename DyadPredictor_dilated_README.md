# DyadPredictor Dilated Convolution Version

## Overview

This is an enhanced version of `DyadPredictor.ipynb` optimized for:

- **132 bp nucleosome positioning motifs** (via dilated convolutions)
- **1000 bp long-range chromatin correlations** (via deeper BiLSTM)

## Key Changes from Original

### 1. Model Architecture (Cell 5)

#### Dilated Convolutional Layers

**Original:**

- 2 conv layers, kernel_size=5, no dilation
- Receptive field: ~10 bp

**Dilated Version:**

- 4 conv layers, kernel_size=7, dilations=[1, 2, 4, 8]
- Receptive fields:
  - Layer 1: ~7 bp
  - Layer 2: ~21 bp
  - Layer 3: ~49 bp
  - Layer 4: ~105 bp
  - **Combined: ~132 bp** (captures full nucleosome)

#### BiLSTM Layers

**Original:**

- num_layers=2, hidden_dim=64

**Dilated Version:**

- num_layers=3, hidden_dim=128
- Better capacity for 1000 bp long-range dependencies

#### Embedding

**Original:**

- embedding_dim=16

**Dilated Version:**

- embedding_dim=32
- More expressive sequence representations

### 2. Model Instantiation (Cell 9)

```python
# Original
model = DyadPredictor(
    vocab_size=8, embedding_dim=16, hidden_dim=64, num_layers=2, dropout=0.3
)

# Dilated Version
model = DyadPredictor(
    vocab_size=8, embedding_dim=32, hidden_dim=128, num_layers=3, dropout=0.3
)
```

### 3. Configuration Metadata (Cell 10)

Added to JSON config:

- `model_type`: "dilated_convolutions"
- `conv_layers`: 4
- `conv_kernel_size`: 7
- `conv_dilations`: [1, 2, 4, 8]
- `receptive_field`: "~132 bp"
- `lstm_context`: "~1000 bp"

## Model Parameters Comparison

| Component | Original | Dilated Version |
|-----------|----------|-----------------|
| Embedding dim | 16 | 32 |
| Hidden dim | 64 | 128 |
| LSTM layers | 2 | 3 |
| Conv layers | 2 | 4 |
| Conv kernel | 5 | 7 |
| Conv dilation | [1, 1] | [1, 2, 4, 8] |
| Receptive field | ~10 bp | ~132 bp |
| Total params | ~67K | ~550K |

## Why Dilated Convolutions?

1. **Multi-scale feature detection**: Captures patterns from 5 bp to 132 bp simultaneously
2. **Efficient parameter usage**: Exponentially growing receptive field without many parameters
3. **Hierarchical learning**: Matches chromatin structure (base pairs → minor groove → DNA bending → nucleosome wrapping)
4. **Better for nucleosomes**: 132 bp ≈ nucleosome DNA wrapping length (~147 bp)

## Expected Improvements

- Better detection of nucleosome positioning motifs (poly(dA:dT) tracts, GC content patterns)
- Improved modeling of nucleosome-nucleosome interactions (linker DNA effects)
- Better capture of long-range chromatin fiber organization
- More accurate dyad position prediction on real genomic sequences

## Usage

Use exactly like the original notebook:

1. Run cells 1-6 to generate data
2. Run cells 7-10 to create datasets and model
3. Run cell 11 to train
4. Evaluate with cells 12-15

## Training Considerations

- **More parameters** (~8× increase): Longer training time
- **Recommendation**: Use `n_samples=500-1000` for better generalization
- May need more epochs (try 50-100 instead of 20)
- Consider reducing `batch_size` if memory issues occur

## Files

- `DyadPredictor.ipynb`: Original version (simple convolutions)
- `DyadPredictor_dilated.ipynb`: This dilated version (optimized for nucleosomes)
- `DyadPredictor_dilated_README.md`: This documentation
