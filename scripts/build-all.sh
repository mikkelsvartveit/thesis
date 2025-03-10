#!/bin/bash
# build-all.sh - Build libraries for all architectures

set -e

# Architectures to build for
ARCHITECTURES=(
  "arm64"
  "riscv64"
  "xtensa"
  # Add more architectures here
  # "ppc64le"
  # "s390x"
  # "mips64"
)

# Libraries to build
LIBRARIES=(
  #  "zlib:1.3"
  #  "libjpeg-turbo:3.1.0"
  #  "libxml2:2.14"
  #  "libpng:1.6.47"
  #  "freetype:2.13.3"
  #  "xzutils:1"
  #  "harfbuzz:10.4.0"
  "pcre2:10.45"
)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Build base image if it doesn't exist
if ! docker image inspect cross-compile-base:latest &>/dev/null; then
  echo "Building base Docker image..."
  docker build -t cross-compile-base:latest -f "${PROJECT_ROOT}/Dockerfile.base" "${PROJECT_ROOT}"
fi

# Build architecture-specific images
for arch in "${ARCHITECTURES[@]}"; do
  if ! docker image inspect "cross-compile-${arch}:latest" &>/dev/null; then
    echo "Building Docker image for ${arch}..."
    docker build -t "cross-compile-${arch}:latest" \
      -f "${PROJECT_ROOT}/architectures/Dockerfile.${arch}" "${PROJECT_ROOT}"
  fi
done

chmod +x "${PROJECT_ROOT}/scripts/"*
# Build each library for each architecture
for lib_info in "${LIBRARIES[@]}"; do
  # Parse library name and version
  IFS=':' read -r lib_name lib_version <<< "${lib_info}"
  echo "Processing ${lib_name} ${lib_version}..."
  
  for arch in "${ARCHITECTURES[@]}"; do
    echo "Building ${lib_name} ${lib_version} for ${arch}..."
    
    # Create output directory
    mkdir -p "${PROJECT_ROOT}/output/${arch}"
    
    # Run container to build the library
    docker run --rm \
      -v "${PROJECT_ROOT}/sources:/workspace/sources" \
      -v "${PROJECT_ROOT}/output/${arch}:/workspace/output/${arch}" \
      -v "${PROJECT_ROOT}/patches:/workspace/patches" \
      -v "${PROJECT_ROOT}/toolchains:/workspace/toolchains" \
      -v "${PROJECT_ROOT}/scripts/build-lib.sh:/usr/local/bin/build-lib" \
      -v "${PROJECT_ROOT}/scripts/download-libs.sh:/usr/local/bin/download-libs" \
      "cross-compile-${arch}:latest" "${lib_name}" "${lib_version}"
      
    echo "Completed build of ${lib_name} for ${arch}"
  done
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
      echo "  - ${lib_name} ${lib_version}: SUCCESS"
    else
      echo "  - ${lib_name} ${lib_version}: FAILED (no static library found)"
    fi
  done
done