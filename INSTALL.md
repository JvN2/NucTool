# Installation Guide

## From GitHub (Recommended)

Install directly from GitHub:

```bash
pip install git+https://github.com/JvN2/NucTool.git
```

## Development Installation

For local development with editable install:

```bash
# Clone the repository
git clone https://github.com/JvN2/NucTool.git
cd NucTool

# Install in editable mode
pip install -e .

# Or with optional dependencies for notebooks
pip install -e ".[notebooks]"

# Or with development tools
pip install -e ".[dev]"
```

## Usage After Installation

Once installed, you can import directly:

```python
# Import main classes
from nuctool import ChromatinFiber, Plotter, DyadPredictorLLM

# Or import specific functions
from nuctool import simulate_chromatin_fibers, compute_vanderlick

# Use the modules
fiber = ChromatinFiber(sequence="ATCG"*1000)
fiber.calc_energy_landscape(amplitude=0.05, period=10.0)
```

## Running Notebooks

After installation, you can still run the example notebooks:

```bash
cd notebooks
jupyter notebook
```

The notebooks are set up to work both:

- **When package is installed**: They'll use the installed package
- **When running from source**: They'll add the local `nuctool/` directory to the path

## Verifying Installation

Check that the package is installed correctly:

```python
import nuctool
print(f"nuctool version: {nuctool.__version__}")
print(f"Available exports: {len(nuctool.__all__)} items")

# Test imports
from nuctool import ChromatinFiber, Plotter, DyadPredictorLLM
print("All imports successful!")
```

## Troubleshooting

### PyTorch Installation

If you encounter issues with PyTorch, install it separately first:

```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Then install nuctool:

```bash
pip install git+https://github.com/JvN2/NucTool.git
```

### Genomepy Data

Some functions require genome data. On first use:

```python
from nuctool import fetch_chromosome_sequence

# Will auto-download sacCer3 genome if not present
seq = fetch_chromosome_sequence(".genomes/sacCer3/sacCer3.fa", "II")
```
