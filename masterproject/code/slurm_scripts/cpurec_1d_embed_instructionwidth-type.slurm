#!/bin/bash

# Usage: sbatch slurm_scripts/job.slurm src/notebooks/<notebook_name>.ipynb

#SBATCH --account=share-ie-idi
#SBATCH --job-name=cpurec_1dembedinstrw         # Job name
#SBATCH --output=./slurm_out/%x.%j.slurm.out   # Standard output file (%j expands to jobID)
#SBATCH --error=./slurm_out/%x.%j.slurm.err    # Standard error file (%j expands to jobID)
#SBATCH --time=4:00:00              # Time limit (hours:min:sec)
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
for i in {1..10}; do
    python ./main.py --config configs/cpu_rec_testing/1dEmbeddingInstructionWidthType.yml --override "{\"validator\": {\"seed\": $i, \"wandb_project_name\": \"stian-02-27-1dinstrw\"}}"
done
for i in {1..10}; do
    python ./main.py --config configs/cpu_rec_testing/1dEmbeddingInstructionWidthType.yml --override "{\"validator\": {\"seed\": $i, \"wandb_project_name\": \"stian-02-27-1dinstrw\"}, \"testing_data\": {\"params\": {\"feature_csv_path\": \"Andreassen-cpu_rec-features.csv\"}}}"
done

echo "Job finished on $(date)"