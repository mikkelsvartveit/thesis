#!/bin/bash

# Usage: sbatch slurm_scripts/job.slurm src/notebooks/<notebook_name>.ipynb

#SBATCH --job-name=notebook                              # Job name %x
#SBATCH --output=./slurm_out/%x.%j.slurm.out        # Standard output file (%j - jobID, %x - job-name)
#SBATCH --error=./slurm_out/%x.%j.slurm.err         # Standard error file (%j - jobID, %x - job-name)
#SBATCH --time=4:00:00                              # Time limit (hours:min:sec)
#SBATCH --nodes=1                                   # Number of nodes requested
#SBATCH --cpus-per-task=12                          # Number of CPUs per task
#SBATCH --mem=16G                                   # Memory requested
#SBATCH --partition=GPUQ                            # Partition/queue requested
#SBATCH --gres=gpu:a100:1                           # GPU resources requested

module load Python/3.12.3-GCCcore-13.3.0

source ~/thesis/masterproject/code/.venv/bin/activate

echo "Job started on $(date)"
echo "Running on host: $(hostname)"

if [ -z "$1" ]; then
    echo "Error: Notebook filename not provided"
    echo "Usage: sbatch job.slurm <notebook_filename.ipynb>"
    exit 1
fi

echo "Running notebook: $1"

notebook_path=$(readlink -f "$1")
notebook_dir=$(dirname "$notebook_path")
notebook_filename=$(basename "$notebook_path" .ipynb)

tmpdir=$(mktemp -d)
jupyter nbconvert --to script "$notebook_path" --output-dir $tmpdir --output="$notebook_filename"

cd "$notebook_dir"
pwd
python $tmpdir/"$notebook_filename.py"

echo "Job finished on $(date)"