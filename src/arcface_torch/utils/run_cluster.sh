#!/bin/bash
#SBATCH --job-name=train_biased_arcface             # Name of your job
#SBATCH --output=logs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=logs/%x_%j.err             # Error file
#SBATCH --partition=A100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:3                  # Request 3 GPUs
#SBATCH --cpus-per-task=16            # Request 16 CPU cores
#SBATCH --mem=64G                     # Request 64 GB of memory
#SBATCH --time=12:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate pydev

srun torchrun --nproc_per_node=3 train_v2.py configs/bupt_biased

# Print job completion time
echo "Job finished at: $(date)"