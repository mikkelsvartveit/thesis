#!/bin/bash
# build-all.sh - Build libraries for all architectures

set -e

# Architectures to build for
ARCHITECTURES=(
 #"arm64"
 # "riscv64"
 # "xtensa"
 # "arcompact"
  "m32r"
  #"epiphany"
  # Add more architectures here
  # "ppc64le"
  # "s390x"
  # "mips64"
)

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

# Build base image if it doesn't exist
if ! docker image inspect crosscompiler-base:latest &>/dev/null; then
  echo "Building base Docker image..."
  docker build -t crosscompiler-base:latest -f "${PROJECT_ROOT}/singularity-images/Dockerfile.base" "${PROJECT_ROOT}"
fi
if ! docker image inspect buildcross-base:latest &>/dev/null; then
  echo "Building base Docker image..."
  docker build -t buildcross-base:latest -f "${PROJECT_ROOT}/singularity-images/Dockerfile.buildcross-base" "${PROJECT_ROOT}"
fi

# Build architecture-specific images
for arch in "${ARCHITECTURES[@]}"; do
  if ! docker image inspect "crosscompiler-${arch}:latest" &>/dev/null; then
    echo "Building Docker image for ${arch}..."
    docker build -t "crosscompiler-${arch}:latest" \
      -f "${PROJECT_ROOT}/architectures/Dockerfile.${arch}" "${PROJECT_ROOT}"
  fi
done

chmod +x "${PROJECT_ROOT}/scripts/"*

# Process each architecture
for arch in "${ARCHITECTURES[@]}"; do
  echo "Processing architecture: ${arch}..."
  
  # Create output directory
  mkdir -p "${PROJECT_ROOT}/output/${arch}"
  
  # Build all libraries for this architecture using a one-liner for loop
  echo "Building all libraries for ${arch}..."
  
  # Convert LIBRARIES array to a Bash-compatible string
  LIB_ARRAY_STR=$(printf "'%s' " "${LIBRARIES[@]}")
  
  echo "Spinning up container"
  docker run --rm --entrypoint /bin/bash \
    -v "${PROJECT_ROOT}/sources:/workspace/sources" \
    -v "${PROJECT_ROOT}/output/${arch}:/workspace/output/${arch}" \
    -v "${PROJECT_ROOT}/patches:/workspace/patches" \
    -v "${PROJECT_ROOT}/toolchains:/workspace/toolchains" \
    -v "${PROJECT_ROOT}/scripts/build-lib.sh:/usr/local/bin/build-lib" \
    -v "${PROJECT_ROOT}/scripts/download-libs.sh:/usr/local/bin/download-libs" \
    "crosscompiler-${arch}:latest" \
      -c "for lib_info in ${LIB_ARRAY_STR}; do \
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