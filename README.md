# NucTool-2

Comprehensive toolkit for chromatin fiber simulation, nucleosome positioning analysis, and machine learning-based dyad prediction.

## Project Structure

```
NucTool-2/
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Package configuration
├── README.md                   # This file
│
├── nuctool/                    # Main Python package
│   ├── __init__.py            # Package exports
│   ├── ChromatinFibers.py     # Core chromatin simulation
│   ├── Plotter.py             # Publication-quality figures
│   ├── pycorrelate.py         # Correlation analysis
│   ├── DyadPredictorLLM.py    # ML models for dyad prediction
│   └── README.md              # Module documentation
│
├── notebooks/                  # Jupyter notebooks for analysis and ML
│   ├── SimulateYeastFibers.ipynb     # Chromatin fiber simulation
│   ├── DyadPredictor.ipynb           # Train simple CNN+LSTM model
│   ├── DyadPredictor_dilated.ipynb   # Train dilated CNN+LSTM model
│   ├── Apply_DyadPredictor.ipynb     # Apply pretrained models
│   ├── nuctool2.ipynb                # Legacy analysis tools
│   └── README.md                     # Notebook documentation
│
├── tests/                     # Test files
│   ├── test_chromatin_fiber.py
│   └── test_plotter.ipynb
│
├── data/                      # Data storage
│   └── LLM models/           # Training data (HDF5) and model weights
│
├── figures/                   # Generated plots and visualizations
│
└── docs/                      # Additional documentation
```

## Core Modules (in `nuctool/` package)

### ChromatinFibers.py

Complete framework for chromatin fiber modeling:

- DNA sequence-dependent nucleosome positioning
- Wrapping energy calculations based on dinucleotide periodicity
- Statistical mechanics modeling (Vanderlick formalism)
- Monte Carlo sampling of nucleosome configurations
- DNA methylation pattern simulation
- Footprint analysis from accessibility data

**Key Classes:**

- `ChromatinFiber` - Main simulation and analysis class
- `SimulationParams` - Configuration for batch simulations
- `WrappingEnergyResult` - Energy calculation results
- `MethylationResult` - Methylation simulation output

### Plotter.py

Publication-quality figure creation with automatic formatting:

- Automatic subplot creation and panel labeling (A, B, C...)
- Extract subplot titles and format into captions
- Constrained layout for consistent appearance
- Specialized chromatin fiber visualization

**Main Class:**

- `Plotter` - Modern plotting utility with caption management

### DyadPredictorLLM.py

Neural network models for nucleosome dyad position prediction:

- Simple CNN + BiLSTM architecture
- Dilated CNN + BiLSTM for multi-scale context
- Training pipeline with checkpointing and early stopping
- Easy model loading and application

## Getting Started

### Installation

**From GitHub (recommended):**

```bash
pip install git+https://github.com/JvN2/NucTool.git
```

**For local development:**

```bash
# Clone repository
git clone https://github.com/JvN2/NucTool.git
cd NucTool

# Install in editable mode
pip install -e .

# Or with optional dependencies for notebooks
pip install -e ".[notebooks]"

# Or with development tools
pip install -e ".[dev]"
```

### Quick Example

```python
from nuctool import ChromatinFiber, Plotter, plot_sequence
import matplotlib.pyplot as plt

# Create chromatin fiber
fiber = ChromatinFiber(sequence="ATCG"*1000, start=0)

# Calculate energy landscape
fiber.calc_energy_landscape(
    amplitude=0.05,
    period=10.0,
    chemical_potential=-2.0
)

# Sample configuration and simulate methylation
dyads = fiber.sample_fiber_configuration()
methylation = fiber.calc_methylation(dyads, efficiency=0.7)

# Visualize
plot = Plotter()
plot.new()
plot_sequence(plot.panels[0], fiber, occupancy=True, dyads=True)
plt.show()
```

### Verify Installation

Check that the package is installed correctly:

```python
import nuctool
print(f"nuctool version: {nuctool.__version__}")
from nuctool import ChromatinFiber, Plotter, DyadPredictorLLM
print("All imports successful!")
```

### Running Notebooks

```bash
cd notebooks
jupyter notebook
```

The notebooks work both with the installed package and when running from source.

## Key Features

### Chromatin Simulation

- Sequence-dependent nucleosome positioning
- Thermodynamic equilibrium calculations
- Stochastic configuration sampling
- DNA methylation with nucleosome protection
- ORF (gene) annotation integration

### Machine Learning

- Per-position dyad prediction from DNA sequences
- Two model architectures (simple and dilated convolutions)
- HDF5-based batch training
- Class imbalance handling
- Mixed precision training

### Visualization

- Publication-ready multi-panel figures
- Automatic panel labeling and caption formatting
- Chromatin fiber plotting with energy landscapes
- Footprint visualization from methylation data

## Citation

If you use this toolkit in your research, please cite:

```text
[Citation information to be added]
```

## License

```text
[License information to be added]
```

## Contact

John van Noort  
[Contact information]
