#!/bin/bash
# Script to submit all cross-compiler build jobs
# Place this in ./slurm-scripts/

# First make sure all scripts are generated
./slurm-scripts/generate_build_scripts.sh

# Submit all jobs
echo "Submitting all build jobs..."
cd ./submitscripts
for script in build_*.sh; do
    echo "Submitting $script..."
    sbatch "$script"
done

echo "All jobs submitted. Check status with 'squeue'"