#!/bin/bash
# build-lib.sh - Build a library for a specific architecture

set -e

# Default values
LIB_NAME=${1:-"zlib"}
LIB_VERSION=${2:-"1.3"}
BUILD_TYPE=${3:-"Release"}

SOURCES_DIR="/workspace/sources"
BUILD_DIR="/workspace/output/${ARCH}/${LIB_NAME}/build"
OUTPUT_DIR="/workspace/output/${ARCH}/${LIB_NAME}/install"
ARCH_PATCH_DIR="/workspace/patches/${LIB_NAME}/${ARCH}"
LIB_PATCH_DIR="/workspace/patches/${LIB_NAME}"

# Print build information
echo "Building ${LIB_NAME} ${LIB_VERSION} for ${ARCH} architecture"
echo "Using toolchain: ${TOOLCHAIN_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

# Check if the library source exists
if [ ! -d "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}" ]; then
  echo "ERROR: Source directory ${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION} not found!"
  echo "Please manually download and extract the source code before building."
  exit 1
fi

echo "Found source directory: ${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"

# Create build directory
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Apply patches if they exist
if [ -d "${ARCH_PATCH_DIR}" ]; then
  echo "Checking and applying patches from ${ARCH_PATCH_DIR}..."
  for patch in "${ARCH_PATCH_DIR}"/*.patch; do
    if [ -f "${patch}" ]; then
      echo "Processing patch: $(basename ${patch})"
      cd "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"
      
      # Check if patch can be applied in reverse - if yes, it's already applied
      if patch -R --dry-run --quiet -p1 < "${patch}"; then
        echo "  Patch already applied, skipping."
      else
        echo "  Applying patch..."
        patch -p1 < "${patch}"
      fi
    fi
  done
fi

# Apply patches if they exist
if [ -d "${LIB_PATCH_DIR}" ]; then
  echo "Checking and applying patches from ${LIB_PATCH_DIR}..."
  for patch in "${LIB_PATCH_DIR}"/*.patch; do
    if [ -f "${patch}" ]; then
      echo "Processing patch: $(basename ${patch})"
      cd "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"
      
      # Check if patch can be applied in reverse - if yes, it's already applied
      if patch -R --dry-run --quiet -p1 < "${patch}"; then
        echo "  Patch already applied, skipping."
      else
        echo "  Applying patch..."
        patch -p1 < "${patch}"
      fi
    fi
  done
fi

# Build using CMake
cd "${BUILD_DIR}"
echo "Configuring build with CMake..."
cmake -G "Ninja" \
      -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DLIBXML2_WITH_ICONV=OFF \
      -DLIBXML2_WITH_PYTHON=OFF \
      -DBOOST_RUNTIME_LINK=STATIC \
      "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"

echo "Building ${LIB_NAME}..."
cmake --build . -j$(nproc)

echo "Installing ${LIB_NAME} to ${OUTPUT_DIR}..."
cmake --install .

echo "Build completed successfully!"
echo "Library installed to ${OUTPUT_DIR}"

# Print information about the built library
echo "Library information:"
if [ -f "${OUTPUT_DIR}/lib/libz.a" ]; then
  ${CROSS_COMPILE}readelf -h "${OUTPUT_DIR}/lib/libz.a" | grep -E "Class|Machine|Version"
fi

# Return to workspace
cd /workspace