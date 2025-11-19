# Notebooks

This directory contains Jupyter notebooks for chromatin fiber simulation, analysis, and machine learning.

## Notebooks

### Core Analysis

- **`SimulateYeastFibers.ipynb`** - Simulate yeast chromatin fibers with nucleosome positioning and methylation patterns
- **`nuctool2.ipynb`** - Legacy chromatin analysis tools and utilities

### Machine Learning - Dyad Prediction

- **`DyadPredictor.ipynb`** - Complete training pipeline for simple CNN+BiLSTM architecture
  - Includes data simulation, model definition, training loop, and evaluation
  - Full implementation details visible in notebook
- **`DyadPredictor_dilated.ipynb`** - Complete training pipeline for dilated CNN+BiLSTM architecture
  - Uses multi-scale dilated convolutions for better long-range context (~147bp nucleosomes)
  - See [detailed documentation](DyadPredictor_dilated_README.md)
- **`Apply_DyadPredictor.ipynb`** - Apply pre-trained models without retraining
  - Load saved models and use for prediction on new data
  - No training code, just inference

Note: All training notebooks use the `DyadPredictorLLM` class (from `nuctool/`) internally for some utilities, but primarily show the full PyTorch implementation details.

## Usage

All notebooks automatically add the `../nuctool/` directory to Python's path, so they can import the main modules:

- `ChromatinFibers` - Core chromatin simulation and analysis
- `Plotter` - Publication-quality figure creation
- `pycorrelate` - Correlation analysis utilities
- `DyadPredictorLLM` - Machine learning model class

The notebooks work both with the installed `nuctool` package and when running from source.

### Running Notebooks

From the project root:

```bash
cd notebooks
jupyter notebook
```

Or from within VS Code, simply open any `.ipynb` file.

### Import Structure

Each notebook starts with:

```python
import sys
from pathlib import Path

# Add nuctool directory to path for imports
nuctool_path = Path.cwd().parent / 'nuctool'
if str(nuctool_path) not in sys.path:
    sys.path.insert(0, str(nuctool_path))

# Now you can import from nuctool package
from ChromatinFibers import ChromatinFiber
from Plotter import Plotter
```

## Data Files

Training data and models are stored in:

- `../data/LLM models/` - HDF5 simulation data and PyTorch model files

Output figures are saved to:

- `../figures/` - Generated plots and visualizations

## Related Directories

- `../nuctool/` - Contains all Python modules (`ChromatinFibers.py`, `Plotter.py`, `DyadPredictorLLM.py`, etc.)
- `../tests/` - Contains `test_plotter.ipynb` for testing plotting utilities
