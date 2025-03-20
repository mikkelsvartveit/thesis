#!/bin/bash
# build-all.sh - Build libraries for all architectures

set -e

# Architectures to build for
SIF_IMAGES="./slurm-scripts/cross-compiler-images/*.sif"

# Libraries to build
LIBRARIES=(
  "zlib:1.3"
  #"libxml2:2.14"
  # "libjpeg-turbo:3.1.0"
  # "libpng:1.6.47"
  # "freetype:2.13.3"
  # "xzutils:1"
  # "pcre2:10.45"
  # "libyaml:0.2.5"
  # "libwebp:1.5.0"
  # C++ required
  #"jsoncpp:1.9.6"
  #"harfbuzz:10.4.0"
)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

chmod +x "${PROJECT_ROOT}/scripts/"*

# Process each architecture
for SIF_image in $SIF_IMAGES; do
  # Extract architecture name from SIF image filename (cross-compiler-ARCH.sif)
  arch=$(basename "${SIF_image}" .sif | sed 's/cross-compiler-//')
  
  echo "Processing architecture: ${arch}..."
  
  # Create output directory
  mkdir -p "${PROJECT_ROOT}/output/${arch}"
  
  # Build all libraries for this architecture using a one-liner for loop
  echo "Building all libraries for ${arch}..."
  
  # Convert LIBRARIES array to a Bash-compatible string
  LIB_ARRAY_STR=$(printf "'%s' " "${LIBRARIES[@]}")
  
  echo "Spinning up Singularity container"
  singularity exec \
    --bind "${PROJECT_ROOT}/sources:/workspace/sources" \
    --bind "${PROJECT_ROOT}/output/${arch}:/workspace/output/${arch}" \
    --bind "${PROJECT_ROOT}/patches:/workspace/patches" \
    --bind "${PROJECT_ROOT}/toolchains:/workspace/toolchains" \
    --bind "${PROJECT_ROOT}/scripts/build-lib.sh:/usr/local/bin/build-lib" \
    --bind "${PROJECT_ROOT}/scripts/download-libs.sh:/usr/local/bin/download-libs" \
    "${SIF_image}" \
    bash -c "for lib_info in ${LIB_ARRAY_STR}; do \
      IFS=':' read -r lib_name lib_version <<< \"\${lib_info}\"; \
      build-lib \"\${lib_name}\" \"\${lib_version}\" || echo \"====== Failed to build ${lib_name} ======\"; \
    done"
  
  echo "Completed all builds for ${arch}"
done

echo "All builds completed successfully!"

# Print summary
echo -e "\nBuild Summary:"
for arch in "${ARCHITECTURES[@]}"; do
  echo "Architecture: ${arch}"
  for lib_info in "${LIBRARIES[@]}"; do
    IFS=':' read -r lib_name lib_version <<< "${lib_info}"
    lib_dir="${PROJECT_ROOT}/output/${arch}/${lib_name}/install/lib"
    # Check if any *.a file exists in the directory
    if [ -n "$(find "${lib_dir}" -name "*.a" -print -quit 2>/dev/null)" ]; then
      echo "  - ${lib_name} ${lib_version}: SUCCESS $(du -sh ${lib_dir})"
    else
      echo "  - ${lib_name} ${lib_version}: FAILED (no static library found)"
    fi
  done
done