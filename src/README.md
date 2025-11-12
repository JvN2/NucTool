# DyadPredictorLLM - Nucleosome Dyad Position Prediction

A comprehensive Python class for training and applying neural network models to predict nucleosome dyad positions from DNA sequences.

## Features

- **Two Model Architectures**:
  - **Simple**: Convolutional layers + Bidirectional LSTM
  - **Dilated**: Multi-scale dilated convolutions + Bidirectional LSTM (better for capturing nucleosome-scale patterns ~147bp)

- **Easy-to-Use API**:
  - Initialize new models or load pre-trained ones from JSON configs
  - Train on HDF5 data files with automatic checkpointing
  - Apply models to predict dyad positions
  - Visualize model architecture

- **Production-Ready**:
  - Handles class imbalance with weighted loss
  - Automatic mixed precision training (AMP)
  - Early stopping and learning rate scheduling
  - Efficient HDF5 data loading
  - Multi-GPU support

## Quick Start

### 1. Create and Train a New Model

```python
from DyadPredictorLLM import DyadPredictorLLM

# Initialize model
predictor = DyadPredictorLLM()
predictor.init_model(
    model_type='simple',  # or 'dilated'
    embedding_dim=16,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
)

# Train on HDF5 data
predictor.train(
    data_filename='data/LLM models/test.h5',
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    patience=5
)

# Plot training history
predictor.plot_training_history()
```

### 2. Load a Pre-Trained Model

```python
predictor = DyadPredictorLLM()
predictor.load_from_json('data/LLM models/test_15000.json')
# Weights automatically loaded from corresponding .pt file
```

### 3. Apply Model to Predict Dyads

```python
# Get predictions for a specific sequence
result = predictor.apply(
    data_filename='data/LLM models/test.h5',
    index=0,
    threshold=0.3,
    return_dict=True
)

print(f"Predicted dyads: {result['predicted_dyads']}")
print(f"True dyads: {result['true_dyads']}")
print(f"Probabilities shape: {result['dyad_probabilities'].shape}")
```

### 4. Visualize Model Architecture

```python
predictor.visualize_model(sequence_length=100)
```

## Model Architectures

### Simple Model (Lighter, Faster)

```
Input (DNA sequence, 0-7 encoded)
  ↓
Embedding Layer (vocab_size=8, embedding_dim=16)
  ↓
Conv1D (kernel=5, hidden_dim=64)
  ↓
Conv1D (kernel=5, hidden_dim=64)
  ↓
Bidirectional LSTM (hidden_dim=64, num_layers=2)
  ↓
Linear → Sigmoid
  ↓
Output (per-position dyad probabilities)
```

### Dilated Model (More Powerful, Captures Multi-Scale Features)

```
Input (DNA sequence, 0-7 encoded)
  ↓
Embedding Layer (vocab_size=8, embedding_dim=32)
  ↓
Dilated Conv1D (kernel=7, dilation=1, hidden_dim=128)
  ↓
Dilated Conv1D (kernel=7, dilation=2, hidden_dim=128)
  ↓
Dilated Conv1D (kernel=7, dilation=4, hidden_dim=128)
  ↓
Dilated Conv1D (kernel=7, dilation=8, hidden_dim=128)
  ↓  [Receptive field ~132 bp, captures nucleosome-scale patterns]
Bidirectional LSTM (hidden_dim=128, num_layers=3)
  ↓
Linear → Sigmoid
  ↓
Output (per-position dyad probabilities)
```

## API Reference

### Class: `DyadPredictorLLM`

#### Methods

**`init_model(model_type='simple', ...)`**

- Initialize a new model
- `model_type`: 'simple' or 'dilated'
- `embedding_dim`: Embedding dimension (16 for simple, 32 for dilated)
- `hidden_dim`: Hidden layer dimension (64 for simple, 128 for dilated)
- `num_layers`: Number of LSTM layers
- `dropout`: Dropout probability
- `conv_dilations`: Dilation rates for dilated model (default: (1,2,4,8))
- `conv_kernel_size`: Kernel size for dilated convolutions

**`load_from_json(json_filename)`**

- Load model configuration and weights from JSON file
- Automatically loads corresponding .pt file with weights

**`train(data_filename, ...)`**

- Train the model on HDF5 data
- `data_filename`: Path to HDF5 file
- `model_filename`: Where to save weights (auto-generated if None)
- `epochs`: Maximum training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `patience`: Early stopping patience
- `max_batches_per_epoch`: Limit batches per epoch (for fast iteration)
- `use_amp`: Use automatic mixed precision (CUDA only)

**`apply(data_filename, index, threshold=0.3, return_dict=False)`**

- Apply model to predict dyad positions
- `data_filename`: Path to HDF5 file
- `index`: Sample index in file
- `threshold`: Probability threshold for calling peaks
- `return_dict`: If True, return full dictionary with all outputs
- Returns: Dictionary with 'dyad_probabilities', 'predicted_dyads', 'true_dyads', 'encoded_sequence'

**`visualize_model(sequence_length=100, save_path=None)`**

- Visualize model architecture and data flow
- `sequence_length`: Example sequence length for visualization
- `save_path`: Optional path to save figure

**`plot_training_history(save_path=None)`**

- Plot training and validation loss curves

## Data Format

The class expects HDF5 files in the format produced by `ChromatinFibers.simulate_chromatin_fibers()`:

- DNA sequences encoded as integers 0-7
- Dyad positions as integer arrays
- Sequence metadata

## Configuration Files

Model configurations are saved as JSON files with the following structure:

```json
{
  "llm": {
    "model_type": "simple",
    "vocab_size": 8,
    "embedding_dim": 16,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "pos_weight": 45.67,
    "learning_rate": 0.001,
    "weight_decay": 1e-05,
    "batch_size": 32,
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau",
    "n_train_samples": 10500,
    "n_val_samples": 2250,
    "n_test_samples": 2250
  }
}
```

## Examples

See `UseDyadPredictorLLM.ipynb` for comprehensive examples including:

- Creating and training models
- Loading pre-trained models
- Applying models to predict dyads
- Batch prediction across multiple samples
- Calculating metrics (precision, recall, F1)
- Visualizing predictions

## Requirements

```
numpy
torch
matplotlib
tqdm
ChromatinFibers  # Custom module for simulation
```

Optional (for metrics):

```
scikit-learn
```

## Performance Notes

- **Simple Model**: ~200K parameters, faster training, good for most tasks
- **Dilated Model**: ~800K parameters (4x larger), better for capturing nucleosome-scale patterns
- **Training Time**: ~2-5 minutes per epoch on GPU (RTX 3090) for 15K samples
- **Memory**: ~2GB GPU memory for batch_size=32

## Tips

1. **For Kaggle**: Set `max_batches_per_epoch` to limit training time
2. **For Production**: Use the dilated model for best accuracy
3. **For Fast Prototyping**: Use the simple model with `max_batches_per_epoch=100`
4. **Threshold Tuning**: Default 0.3 works well, but try 0.2-0.4 range for your data

## License

MIT License - See main project LICENSE file

## Citation

If you use this code in your research, please cite:

```
[Your paper citation here]
```

## Contact

For questions or issues, please open an issue on GitHub.
