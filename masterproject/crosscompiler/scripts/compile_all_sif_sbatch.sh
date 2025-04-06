#!/bin/bash

# Get all SIF images
SIF_IMAGES="./singularity-images/crosscompiler-images/*.sif"

# Array to store job IDs
declare -a JOB_IDS

# Launch a job for each SIF image
for SIF_image in $SIF_IMAGES; do
    # Extract architecture name from SIF image filename
    arch=$(basename "${SIF_image}" .sif | sed 's/crosscompiler-//')
    
    # Submit the job and capture the job ID
    JOB_ID=$(sbatch --parsable \
        --account="share-ie-idi" \
        --time="00:10:00" \
        --nodes="1" \
        --cpus-per-task="8" \
        --mem="16G" \
        --partition="CPUQ" \
        --job-name="${arch}" \
        --output="./slurm-logs/compilelogs/${arch}/%j.out" \
        --error="./slurm-logs/compilelogs/${arch}/%j.err" \
        --wrap="./scripts/compile_all_sif.sh $arch")
    
    # Store the job ID
    JOB_IDS+=($JOB_ID)
    
    echo "Submitted job $JOB_ID for (arch: $arch)"
done

# Create a dependency string for all jobs
DEPEND_STR=$(IFS=: ; echo "${JOB_IDS[*]}")

# Submit the dataset generation job with dependency on all previous jobs
sbatch \
    --account="share-ie-idi" \
    --time="00:10:00" \
    --nodes="1" \
    --cpus-per-task="8" \
    --mem="16G" \
    --partition="CPUQ" \
    --job-name="dataset-generation" \
    --output="./slurm-logs/compilelogs/dataset-generation-%j.out" \
    --error="./slurm-logs/compilelogs/dataset-generation-%j.err" \
    --dependency=afterany:$DEPEND_STR \
    --wrap="./scripts/result-gen/generate_dataset.sh"

echo "Dataset generation job submitted, will run after all build jobs complete"