#!/bin/bash

# Usage: sbatch slurm_scripts/job.slurm src/notebooks/<notebook_name>.ipynb

#SBATCH --account=share-ie-idi
#SBATCH --job-name=resnet-embedding-endianness         # Job name
#SBATCH --output=./slurm_out/%x.%j.slurm.out   # Standard output file (%j expands to jobID)
#SBATCH --error=./slurm_out/%x.%j.slurm.err    # Standard error file (%j expands to jobID)
#SBATCH --time=2:00:00              # Time limit (hours:min:sec)
#SBATCH --nodes=1                   # Number of nodes requested
#SBATCH --cpus-per-task=12          # Number of CPUs per task
#SBATCH --mem=16G                   # Memory requested
#SBATCH --partition=GPUQ            # Partition/queue requested
#SBATCH --gres=gpu:a100:1           # GPU resources requested

module load Python/3.12.3-GCCcore-13.3.0

source ~/thesis/masterproject/code/.venv/bin/activate

echo "Job started on $(date)"
echo "Running on host: $(hostname)"

cd src
python ./main.py --config configs/EmbeddingResNet50.yml --override "{\"validator\": {\"seed\": 1, \"wandb_project_name\": \"mikkel-02-26\"}}"

echo "Job finished on $(date)"