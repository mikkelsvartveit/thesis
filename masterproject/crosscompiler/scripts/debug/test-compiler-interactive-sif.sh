arch=$1

echo "Running container for $arch"
singularity shell \
      --bind "./:/workspace" \
      "./singularity-images/crosscompiler-images/crosscompiler-${arch}.sif"