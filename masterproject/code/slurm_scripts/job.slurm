#!/bin/bash

#SBATCH --job-name=my_job           # Job name
#SBATCH --output=job_%j.slurm.out         # Standard output file (%j expands to jobID)
#SBATCH --error=job_%j.slurm.err          # Standard error file (%j expands to jobID)
#SBATCH --time=1:00:00              # Time limit (hours:min:sec)
#SBATCH --nodes=1                   # Number of nodes requested
#SBATCH --cpus-per-task=12          # Number of CPUs per task
#SBATCH --mem=16G                   # Memory requested
#SBATCH --partition=GPUQ            # Partition/queue requested
#SBATCH --gres=gpu:a100:1           # GPU resources requested

module load Python/3.12.3-GCCcore-13.3.0

source ~/thesis/masterproject/code/.venv/bin/activate
echo "which python: $(which python)"

echo "Job started on $(date)"
echo "Running on host: $(hostname)"


python job.py


echo "Job finished on $(date)"