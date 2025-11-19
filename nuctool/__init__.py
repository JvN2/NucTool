"""
NucTool - Chromatin fiber simulation and nucleosome dyad prediction toolkit

Main modules:
    ChromatinFibers - Chromatin simulation and analysis
    Plotter - Publication-quality figure creation
    DyadPredictorLLM - Neural network models for dyad prediction
    pycorrelate - Correlation analysis utilities
"""

from .ChromatinFibers import (
    ChromatinFiber,
    SimulationParams,
    WrappingEnergyResult,
    MethylationResult,
    simulate_chromatin_fibers,
    read_simulation_results,
    convert_to_footprints,
    compute_vanderlick,
    fetch_chromosome_sequence,
)

from .Plotter import (
    Plotter,
    plot_sequence,
    plot_footprints,
)

from .DyadPredictorLLM import (
    DyadPredictorLLM,
    SimpleDyadPredictor,
    DilatedDyadPredictor,
    DyadDataset,
)

from .pycorrelate import pcorrelate

__version__ = "0.1.0"

__all__ = [
    # ChromatinFibers
    "ChromatinFiber",
    "SimulationParams",
    "WrappingEnergyResult",
    "MethylationResult",
    "simulate_chromatin_fibers",
    "read_simulation_results",
    "convert_to_footprints",
    "compute_vanderlick",
    "fetch_chromosome_sequence",
    # Plotter
    "Plotter",
    "plot_sequence",
    "plot_footprints",
    # DyadPredictorLLM
    "DyadPredictorLLM",
    "SimpleDyadPredictor",
    "DilatedDyadPredictor",
    "DyadDataset",
    # pycorrelate
    "pcorrelate",
]

__all__ = [
    "DyadPredictorLLM",
    "SimpleDyadPredictor",
    "DilatedDyadPredictor",
    "DyadDataset",
]
__version__ = "1.0.0"
