#!/bin/bash
set -e
# Script to submit all cross-compiler build jobs
# Place this in ./slurm-logs/

# First make sure all scripts are generated
./scripts/generate_build_scripts.sh

# Submit all jobs
echo "Submitting all build jobs..."
SCRIPT_DIR="./singularity-images/submit-scripts"
for script in $SCRIPT_DIR/build_*.slurm; do
    echo "Submitting $script..."
    sbatch "$script"
done

echo "All jobs submitted. Check status with 'squeue'"