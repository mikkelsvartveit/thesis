#!/bin/bash
set -e

# Create needed directories
mkdir -p ./singularity-images/submit-scripts
mkdir -p ./slurm-logs/logs
mkdir -p ./singularity-images/crosscompiler-images
mkdir -p ./singularity-images/crosscompiler-definitions

# List of supported architectures and their targets as colon-separated items
# Format: "arch:target"
ARCH_TARGET_LIST=(
    "arc:arc-unknown-linux-gnu" # ARCompact v2 basicallly (ARCv2)
    "arceb:arceb-unknown-elf" 
    "bfin:bfin-unknown-linux-uclibc"
    "bpf:bpf-unknown-none"
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
    "loongarch64:loongarch64-unknown-linux-gnu"
    "m32r:m32r-unknown-elf"
    "m68k-elf:m68k-unknown-elf"
    "mcore:mcore-unknown-elf"
    "mcoreeb:mcore-unknown-elf" 
    "microblaze:microblaze-unknown-linux-gnu"
    "microblazeel:microblazeel-unknown-linux-gnu" 
    "mmix:mmix-knuth-mmixware"
    "mn10300:mn10300-unknown-elf"
    "moxie:moxie-unknown-elf"
    "moxieel:moxie-unknown-elf" 
    "msp430:msp430-unknown-elf"
    "nds32:nds32le-unknown-linux-gnu"
    "nios2:nios2-unknown-linux-gnu"
    "or1k:or1k-unknown-linux-gnu"
    "pru:pru-unknown-elf"
    "rl78:rl78-unknown-elf"
    "rx:rx-unknown-elf"
    "tilegx:tilegx-unknown-linux-gnu"
    "tricore:tricore-unknown-elf"
    "v850:v850e-uclinux-uclibc"
    "visium:visium-unknown-elf"
    "xstormy16:xstormy16-unknown-elf"
    "xtensa:xtensa-unknown-linux-uclibc"
)

echo "Generating Singularity definition files..."
rm -rf ./singularity-images/crosscompiler-definitions/*
for arch_target in "${ARCH_TARGET_LIST[@]}"; do
    arch=$(echo "$arch_target" | cut -d':' -f1)
    target=$(echo "$arch_target" | cut -d':' -f2)
    def_file="./singularity-images/crosscompiler-definitions/${arch}.def"
    
    echo "  Generating definition file for ${arch}..."
    
    cat > "${def_file}" << EOF
Bootstrap: localimage
From: ./singularity-images/buildcross-base.sif
Stage: builder

%environment
    export ARCH=${arch}
    export TARGET=${target}
    export TOOLCHAIN_FILE="/workspace/toolchains/${arch}.cmake"

%files
    ./buildcross.sh /opt/buildcross/scripts/buildcross.sh

%post
    ARCH=${arch}
    TARGET=${target}

    chmod +x /opt/buildcross/scripts/buildcross.sh
    (build.sh -j\$(nproc) \$ARCH) || \\
        (echo "Failed with precompiled host-gcc..." && \\
        echo "Removing precompiled host-gcc and trying again" && \\
        rm -rf \$BUILDCROSS_HOST_TOOLS/* && build.sh -j\$(nproc) \$ARCH)

# Second stage
Bootstrap: localimage
From: ./singularity-images/crosscompiler-base.sif
Stage: final

%files from builder
    /cross-${arch} /cross-${arch}

%environment
    export ARCH=${arch}
    export TARGET=${target}
    export CROSS_PREFIX="/cross-${arch}" 
    export OUTPUT_PREFIX="/workspace/output/${arch}" 
    export SOURCE_PREFIX="/workspace/sources" 
    export PATH="\${CROSS_PREFIX}/bin:\${PATH}"
    export CC="${target}-gcc"
    export AR="${target}-ar"
    export RANLIB="${target}-ranlib"

%post
    echo "Container built for ${arch} architecture"
EOF
done

echo "Generating SLURM submission scripts..."
rm -rf ./singularity-images/submit-scripts/*

for arch_target in "${ARCH_TARGET_LIST[@]}"; do
    arch=$(echo "$arch_target" | cut -d':' -f1)
    target=$(echo "$arch_target" | cut -d':' -f2)
    submit_script="./singularity-images/submit-scripts/build_${arch}.slurm"
    
    echo "  Generating slurm submission script for ${arch}..."
    
    cat > "${submit_script}" << EOF
#!/bin/bash
#SBATCH --job-name=${arch}_crosscompiler
#SBATCH --output=./slurm-logs/imagebuildlogs/${arch}/%j.log
#SBATCH --error=./slurm-logs/imagebuildlogs/${arch}/%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=16
#SBATCH --account="share-ie-idi"
#SBATCH --partition="CPUQ"
#SBATCH --nodelist="idun-03-[01-48],idun-04-[01-36],idun-05-[01-32]"


# Create log directory
mkdir -p ./slurm-logs/imagebuildlogs/${arch}

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting build for ${arch} (${target})"
start_seconds=\$(date +%s)

# Build the Singularity container using the definition file
echo "Building Singularity container..."
mkdir -p ./singularity-images/crosscompiler-images/
singularity build --fakeroot ./singularity-images/crosscompiler-images/crosscompiler-${arch}.sif ./singularity-images/crosscompiler-definitions/${arch}.def
exit_status=\$?

# Calculate time difference
end_seconds=\$(date +%s)
diff=\$((end_seconds - start_seconds))

# Convert to minutes and seconds
minutes=\$((diff / 60))
seconds=\$((diff % 60))
build_time="\$minutes m \$seconds s"

if [ \$exit_status -eq 0 ]; then
    echo "Singularity container for ${arch} built successfully"
    echo "[SUCCESS] ${arch} \$build_time" >> ./slurm-logs/imagebuildlogs/build-summary.txt
else
    echo "Failed to build Singularity container for ${arch}, exiting..."
    echo "[FAILED] ${arch} \$build_time" >> ./slurm-logs/imagebuildlogs/build-summary.txt && exit 1
fi

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Build for ${arch} completed in \$minutes minutes and \$seconds seconds"
EOF

    chmod +x "${submit_script}"
done