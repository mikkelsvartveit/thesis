#!/bin/bash
set -e
# Master script to generate Slurm submission scripts for all architectures using Singularity
# Place this in ./slurm-scripts/

# Create needed directories
mkdir -p ./slurm-scripts/submitscripts
mkdir -p ./slurm-scripts/logs
mkdir -p ./slurm-scripts/cross-compiler-images
mkdir -p ./slurm-scripts/singularity-definitions

# Create base images for building and cross-compiling
# buildcross-base.tar and cross-compile-base.tar should be created beforehand using 
#   docker save -o
if [ ! -f ./base-images/buildcross-base.sif ]; then
    singularity build ./base-images/buildcross-base.sif docker-archive://./base-images/buildcross-base.tar
else
    echo "buildcross-base.sif file already exists, skipping build"
fi

if [ ! -f ./base-images/cross-compile-base.sif ]; then
    singularity build ./base-images/cross-compile-base.sif docker-archive://./base-images/cross-compile-base.tar
else
    echo "cross-compile-base.sif file already exists, skipping build"
fi

# List of supported architectures and their targets as colon-separated items
# Format: "arch:target"
# You can modify this list as needed
ARCH_TARGET_LIST=(
    "bfin:bfin-unknown-linux-uclibc"
    "c6x:c6x-unknown-uclinux"
    "cr16:cr16-unknown-elf"
    "cris:crisv32-unknown-linux-uclibc"
    "csky:csky-unknown-linux-gnu"
    "epiphany:epiphany-unknown-elf"
    "fr30:fr30-unknown-elf"
    "frv:frv-unknown-linux-uclibc"
    "ft32:ft32-unknown-elf"
    "h8300:h8300-unknown-linux-uclibc"
    "iq2000:iq2000-unknown-elf"
    "kvx:kvx-unknown-linux-uclibc"
    "lm32:lm32-uclinux-uclibc"
    "m32r:m32r-unknown-elf"
    "m68k-elf:m68k-unknown-elf"
    "m68k-uclibc:m68k-unknown-linux-uclibc"
    "mcore:mcore-unknown-elf"
    "mmix:mmix-knuth-mmixware"
    "mn10300:mn10300-unknown-elf"
    "moxie:moxie-unknown-elf"
    "msp430:msp430-unknown-elf"
    "nds32:nds32le-unknown-linux-gnu"
    "pdp11:pdp11-unknown-aout"
    "pru:pru-unknown-elf"
    "rl78:rl78-unknown-elf"
    "rx:rx-unknown-elf"
    "tilegx:tilegx-unknown-linux-gnu"
    "tricore:tricore-unknown-elf"
    "v850:v850e-uclinux-uclibc"
    "visium:visium-unknown-elf"
    "xstormy16:xstormy16-unknown-elf"
    "xtensa:xtensa-unknown-linux-uclibc"
    # "nvptx:nvptx-none" nvidia parallel thread execution, jitcompiled by nvidia driver, not proper arch
)

# First, generate all Singularity definition files
echo "Generating Singularity definition files..."
for arch_target in "${ARCH_TARGET_LIST[@]}"; do
    # Split the string by colon
    arch=$(echo "$arch_target" | cut -d':' -f1)
    target=$(echo "$arch_target" | cut -d':' -f2)
    def_file="./slurm-scripts/singularity-definitions/${arch}.def"
    
    echo "Generating definition file for ${arch}..."
    
    # Create the Singularity definition file
    cat > "${def_file}" << EOF
Bootstrap: localimage
From: ./base-images/buildcross-base.sif
Stage: builder

%environment
    export ARCH=${arch}
    export TARGET=${target}
    export TOOLCHAIN_FILE="/workspace/toolchains/${arch}.cmake"

%post
    ARCH=${arch}
    TARGET=${target}
    build.sh -j\$(nproc) \$ARCH

# Second stage
Bootstrap: localimage
From: ./base-images/cross-compile-base.sif
Stage: final

%files from builder
    /cross-${arch} /cross-${arch}

%environment
    export ARCH=${arch}
    export TARGET=${target}
    export CROSS_PREFIX="/cross-${arch}" 
    export OUTPUT_PREFIX="/workspace/output/${arch}" 
    export SOURCE_PREFIX="/workspace/sources" 
    export TOOLCHAIN_FILE="/workspace/toolchains/${arch}.cmake"
    export PATH="\${CROSS_PREFIX}/bin:\${PATH}"
    export CC="${target}-gcc"
    export AR="${target}-ar"
    export RANLIB="${target}-ranlib"
    export CFLAGS="-g"

%post
    echo "Container built for ${arch} architecture"
EOF
    
    echo "Generated definition file for ${arch}"
done

# Now generate all SLURM submission scripts
echo "Generating SLURM submission scripts..."
rm -rf ./slurm-scripts/submitscripts/*

for arch_target in "${ARCH_TARGET_LIST[@]}"; do
    # Split the string by colon
    arch=$(echo "$arch_target" | cut -d':' -f1)
    target=$(echo "$arch_target" | cut -d':' -f2)
    submit_script="./slurm-scripts/submitscripts/build_${arch}.slurm"
    
    echo "Generating submission script for ${arch}..."
    
    # Create the Slurm submission script
    cat > "${submit_script}" << EOF
#!/bin/bash
#SBATCH --job-name=${arch}_crosscompiler
#SBATCH --output=./slurm-scripts/logs/${arch}/%j.out
#SBATCH --error=./slurm-scripts/logs/${arch}/%j.err
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --account="share-ie-idi"
#SBATCH --partition="CPUQ"

# Create log directory
mkdir -p ./slurm-scripts/logs/${arch}

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting build for ${arch} (${target})"

# Build the Singularity container using the definition file
echo "Building Singularity container..."
mkdir -p ./slurm-scripts/cross-compiler-images/
singularity build --fakeroot ./slurm-scripts/cross-compiler-images/cross-compiler-${arch}.sif ./slurm-scripts/singularity-definitions/${arch}.def

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Build for ${arch} completed"
EOF

    # Make the script executable
    chmod +x "${submit_script}"
    
    # Create the log directory for this architecture
    mkdir -p "./slurm-scripts/logs/${arch}"
    
    echo "Generated script for ${arch}"
done

echo "All Singularity definition files and submission scripts generated."
echo "You can submit the SLURM jobs with:"
echo "cd ./slurm-scripts/submitscripts && for f in build_*.slurm; do sbatch \$f; done"