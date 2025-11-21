#!/bin/bash
#SBATCH --job-name=sim_chromatin
#SBATCH --output=logs/simulation_%j.out
#SBATCH --error=logs/simulation_%j.err
#SBATCH --partition=cpu-short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Chromatin fiber simulation job for SLURM cluster
# Submit with: sbatch scripts/run_simulation.sbatch
# Available partitions: cpu-short (4h), cpu-zen4/cpu-skylake (7d), gpu-short (4h), etc.

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Create logs directory if needed
mkdir -p logs

# Load required modules (adjust for your cluster)
# module load python/3.11
# module load hdf5

# Activate virtual environment if using one
# source venv/bin/activate

# Run simulation
python scripts/run_simulation.py \
    --n_samples 5000 \
    --output "data/LLM models/test_5000.h5" \
    --append

echo ""
echo "Job finished at $(date)"
