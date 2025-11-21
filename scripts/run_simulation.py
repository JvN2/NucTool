#!/usr/bin/env python3
"""
Standalone script to simulate chromatin fibers for dyad prediction training.
Can be run directly or submitted to a SLURM cluster.

Usage:
    python scripts/run_simulation.py --n_samples 5000 --output data/LLM_models/test_5000.h5
    python scripts/run_simulation.py --help
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from nuctool.ChromatinFibers import simulate_chromatin_fibers, SimulationParams, read_simulation_results


def main():
    parser = argparse.ArgumentParser(
        description="Simulate chromatin fibers and save to HDF5 for dyad prediction training"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of chromatin fiber samples to simulate (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/LLM models/simulation.h5",
        help="Output HDF5 file path (default: data/LLM models/simulation.h5)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file if it has fewer samples than requested",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file even if it has enough samples",
    )

    args = parser.parse_args()

    # Resolve output path relative to project root
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Chromatin Fiber Simulation")
    print(f"=" * 60)
    print(f"Project root: {project_root}")
    print(f"Output file:  {output_path}")
    print(f"Samples requested: {args.n_samples}")
    print()

    # Check if file exists and handle append/force logic
    if output_path.exists() and not args.force:
        print(f"File exists: {output_path}")
        existing_params = read_simulation_results(str(output_path))
        existing_samples = existing_params.n_samples
        print(f"Existing samples: {existing_samples}")

        if existing_samples >= args.n_samples:
            print(f"✓ File already contains {existing_samples} samples (>= {args.n_samples} requested)")
            print("Nothing to do. Use --force to regenerate.")
            return 0

        if args.append:
            n_new = args.n_samples - existing_samples
            print(f"Appending {n_new} new samples to reach {args.n_samples} total...")
            simulation_params = existing_params
            simulation_params.n_samples = n_new
            simulate_chromatin_fibers(simulation_params, str(output_path))
        else:
            print(f"File has only {existing_samples} samples but --append not specified.")
            print("Use --append to add more samples, or --force to overwrite.")
            return 1
    else:
        # New simulation
        if args.force and output_path.exists():
            print(f"Overwriting existing file (--force)...")
        print(f"Generating {args.n_samples} new samples...")
        simulation_params = SimulationParams(n_samples=args.n_samples)
        simulate_chromatin_fibers(simulation_params, str(output_path))

    # Verify and report final state
    final_params = read_simulation_results(str(output_path))
    print()
    print(f"✓ Simulation complete!")
    print(f"Final parameters:")
    for key, value in final_params.__dict__.items():
        print(f"  {key}: {value}")
    print()
    print(f"Output saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
