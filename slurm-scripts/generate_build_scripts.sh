#!/bin/bash
# Master script to generate Slurm submission scripts for all architectures
# Place this in ./slurm-scripts/

# Create needed directories
mkdir -p ./slurm-scripts/submitscripts
mkdir -p ./slurm-scripts/logs
mkdir -p ./slurm-scripts/cross-compiler-images

# List of supported architectures and their targets as colon-separated items
# Format: "arch:target"
# You can modify this list as needed
ARCH_TARGET_LIST=(
    "m32r:m32r-unknown-elf"
    "ft32:ft32-unknown-elf"
    "epiphany:epiphany-unknown-elf"
    # Add more architectures as needed
)

# Import base images
echo "Importing base images..."
podman load -i ./base-images/buildcross-base.tar
podman load -i ./base-images/cross-compile-base.tar

# For each architecture, generate a Slurm submission script
for arch_target in "${ARCH_TARGET_LIST[@]}"; do
    # Split the string by colon
    arch=$(echo "$arch_target" | cut -d':' -f1)
    target=$(echo "$arch_target" | cut -d':' -f2)
    submit_script="./slurm-scripts/submitscripts/build_${arch}.slurm"
    
    echo "Generating submission script for ${arch}..."
    
    cat > "${submit_script}" << EOF
#!/bin/bash
#SBATCH --job-name=crosscompile_${arch}
#SBATCH --output=./logs/${arch}/%j.out
#SBATCH --error=./logs/${arch}/%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --account="share-ie-idi"
#SBATCH --partition="CPUQ"

# Create log directory
mkdir -p ./logs/${arch}

echo "Starting build for ${arch} (${target})"

# Build the cross-compiler container
podman build \\
    --build-arg ARCH=${arch} \\
    --build-arg TARGET=${target} \\
    -t cross-compiler-${arch}:latest \\
    -f ./architectures/Dockerfile.template .

# Export the container as a tar
echo "Exporting container to tar file..."
podman save -o ./cross-compiler-images/cross-compiler-${arch}.tar cross-compiler-${arch}:latest

echo "Build for ${arch} completed"
EOF

    # Make the script executable
    chmod +x "${submit_script}"
    
    # Create the log directory for this architecture
    mkdir -p "./logs/${arch}"
    
    echo "Generated script for ${arch}"
done

echo "All submission scripts generated. You can submit them with:"
echo "cd ./slurm-scripts/submitscripts && for f in build_*.sh; do sbatch \$f; done"