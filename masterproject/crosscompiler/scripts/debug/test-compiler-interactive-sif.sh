arch=$1

echo "Running container for $arch"
singularity shell \
      --bind "./sources:/workspace/sources" \
      --bind "./output/${arch}:/workspace/output/${arch}" \
      --bind "./patches:/workspace/patches" \
      --bind "./toolchains:/workspace/toolchains" \
      --bind "./scripts/build-lib.sh:/usr/local/bin/build-lib" \
      --bind "./scripts/download-libs.sh:/usr/local/bin/download-libs" \
      "./singularity-images/crosscompiler-images/cross-compiler-${arch}.sif"