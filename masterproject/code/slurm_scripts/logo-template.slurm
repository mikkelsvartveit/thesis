#!/bin/bash

#SBATCH --account=share-ie-idi
#SBATCH --output=./slurm_out/%j.%x.slurm.out   # Standard output file (%j expands to jobID)
#SBATCH --error=./slurm_out/%j.%x.slurm.err    # Standard error file (%j expands to jobID)
#SBATCH --nodes=1                   # Number of nodes requested
#SBATCH --cpus-per-task=12          # Number of CPUs per task
#SBATCH --mem=16G                   # Memory requested
#SBATCH --partition=GPUQ            # Partition/queue requested
#SBATCH --gres=gpu:a100:1           # GPU resources requested

#SBATCH --job-name=logo
#SBATCH --time=4:00:00
TARGET_FEATURE="endianness"
CONFIG_PATH="configs/logo/Simple1dEmbedding.yml"
NUM_RUNS=20
WANDB_PROJECT_NAME="job-$SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0

source ~/thesis/masterproject/code/.venv/bin/activate

echo "Job started on $(date)"
echo "Running on host: $(hostname)"

cd src
for i in $(seq 1 $NUM_RUNS); do
    python ./main.py --config $CONFIG_PATH --override "{\"target_feature\": \"$TARGET_FEATURE\", \"validator\": {\"seed\": $i, \"wandb_project_name\": \"$WANDB_PROJECT_NAME\"}}"
done

echo "Job finished on $(date)"