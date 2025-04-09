#!/bin/bash

#SBATCH --account=share-ie-idi
#SBATCH --output=./slurm_out/%j.%x.slurm.out   # Standard output file (%j expands to jobID)
#SBATCH --error=./slurm_out/%j.%x.slurm.err    # Standard error file (%j expands to jobID)
#SBATCH --nodes=1                   # Number of nodes requested
#SBATCH --cpus-per-task=12          # Number of CPUs per task
#SBATCH --mem=16G                   # Memory requested
#SBATCH --partition=GPUQ            # Partition/queue requested
#SBATCH --gres=gpu:a100:1           # GPU resources requested

#SBATCH --job-name=cpurec-all
#SBATCH --time=01:00:00
WANDB_PROJECT_NAME="cpurec-all-$SLURM_JOB_ID"
TARGET_FEATURES=("endianness" "instructionwidth_type")

module load Python/3.12.3-GCCcore-13.3.0

source ~/thesis/masterproject/code/.venv/bin/activate

echo "Job started on $(date)"
echo "Running on host: $(hostname)"

run_experiment() {
    target_feature=$1
    config_path=$2
    num_runs=$3

    for i in $(seq 1 $num_runs); do
        python ./main.py --config $config_path --override "{\"target_feature\": \"$target_feature\", \"validator\": {\"seed\": $i, \"wandb_project_name\": \"$WANDB_PROJECT_NAME\"}}"
    done
}

cd src

for target_feature in "${TARGET_FEATURES[@]}"; do
    run_experiment "$target_feature" "configs/cpurec/Simple1d.yml" 5
    run_experiment "$target_feature" "configs/cpurec/Simple1dEmbedding.yml" 5
    run_experiment "$target_feature" "configs/cpurec/Simple2d.yml" 5
    run_experiment "$target_feature" "configs/cpurec/Simple2dEmbedding.yml" 5
    run_experiment "$target_feature" "configs/cpurec/ResNet50.yml" 5
    run_experiment "$target_feature" "configs/cpurec/ResNet50Embedding.yml" 5
done

echo "Job finished on $(date)"