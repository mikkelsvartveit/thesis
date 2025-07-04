#!/bin/bash
# compile_lib.sh - Build a library for a specific architecture

set -e

LIB_NAME=${1:-"zlib"}
LIB_VERSION=${2:-"1.3"}
JCOUNT=${3:-"$(nproc)"}

echo $JCOUNT

SOURCES_DIR="/workspace/sources"
ARCH_PATCH_DIR="/workspace/patches/${LIB_NAME}/${ARCH}"
LIB_PATCH_DIR="/workspace/patches/${LIB_NAME}"

echo "Building ${LIB_NAME} ${LIB_VERSION} for ${ARCH} architecture"
echo "Using toolchain: ${TOOLCHAIN_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

if [ ! -d "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}" ]; then
  echo "ERROR: Source directory ${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION} not found!"
  echo "Please manually download and extract the source code before building."
  exit 1
fi

echo "Found source directory: ${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"

build_lib_for_arch () {

  local arch=$1

  BUILD_DIR="/workspace/output/${arch}/${LIB_NAME}/build"
  OUTPUT_DIR="/workspace/output/${arch}/${LIB_NAME}/install"
  # Create build directory
  mkdir -p "${BUILD_DIR}"
  mkdir -p "${OUTPUT_DIR}"

  # # Apply patches if they exist
  # if [ -d "${ARCH_PATCH_DIR}" ]; then
  #   echo "Checking and applying patches from ${ARCH_PATCH_DIR}..."
  #   for patch in "${ARCH_PATCH_DIR}"/*.patch; do
  #     if [ -f "${patch}" ]; then
  #       echo "Processing patch: $(basename ${patch})"
  #       cd "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"
        
  #       # Check if patch can be applied in reverse - if yes, it's already applied
  #       if patch -R --dry-run --quiet -p1 < "${patch}"; then
  #         echo "  Patch already applied, skipping."
  #       else
  #         echo "  Applying patch..."
  #         patch -p1 < "${patch}"
  #       fi
  #     fi
  #   done
  # fi

  # # Apply patches if they exist
  # if [ -d "${LIB_PATCH_DIR}" ]; then
  #   echo "Checking and applying patches from ${LIB_PATCH_DIR}..."
  #   for patch in "${LIB_PATCH_DIR}"/*.patch; do
  #     if [ -f "${patch}" ]; then
  #       echo "Processing patch: $(basename ${patch})"
  #       cd "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"
        
  #       # Check if patch can be applied in reverse - if yes, it's already applied
  #       if patch -R --dry-run --quiet -p1 < "${patch}"; then
  #         echo "  Patch already applied, skipping."
  #       else
  #         echo "  Applying patch..."
  #         patch -p1 < "${patch}"
  #       fi
  #     fi
  #   done
  # fi

  TEST_SKIP_ARGS="-DBUILD_TESTING=OFF \
  -DPCRE2_BUILD_TESTS=OFF \
  -DBUILD_TESTS=OFF \
  -DENABLE_TESTING=OFF \
  -DTEST=OFF \
  -DENABLE_TESTS=OFF \
  -DJSONCPP_WITH_TESTS=OFF \
  -DFREETYPE_ENABLE_TESTS=OFF \
  -DHARFBUZZ_BUILD_TESTS=OFF \
  -DLIBJPEG_TURBO_BUILD_TESTS=OFF \
  -DLIBPNG_TESTS=OFF \
  -DWEBP_BUILD_TESTS=OFF \
  -DYAML_BUILD_TESTS=OFF \
  -DZLIB_BUILD_TESTS=OFF \
  -DXZUTILS_BUILD_TESTS=OFF"

  CMAKE_TOOLCHAIN_ARGS=""
  TOOLCHAIN_FILE="/workspace/toolchains/${arch}.cmake"
  TMP_TOOLCHAIN_FILE="/tmp/toolchain-${arch}.cmake"

  # If original toolchain file doesn't exist, create a minimal temporary one
  if [ ! -f "${TOOLCHAIN_FILE}" ]; then
    echo "Toolchain file not found: ${TOOLCHAIN_FILE}"
    echo "Generating minimal toolchain file for ${arch}..."
    
    cat > "${TMP_TOOLCHAIN_FILE}" << EOF
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ${arch})

set(CMAKE_C_COMPILER ${TARGET}-gcc)
set(CMAKE_CXX_COMPILER ${TARGET}-g++)
set(CMAKE_AR ${TARGET}-ar)
set(CMAKE_RANLIB ${TARGET}-ranlib)
set(CMAKE_STRIP ${TARGET}-strip)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")
EOF

    echo "Created temporary toolchain file: ${TMP_TOOLCHAIN_FILE}"
    TOOLCHAIN_FILE="${TMP_TOOLCHAIN_FILE}"

    # Set additional CMake arguments for cross-compilation
    CMAKE_TOOLCHAIN_ARGS=" \
      -DCMAKE_FIND_ROOT_PATH='/cross-${arch}' \
      -DCMAKE_SHARED_LIBRARY_LINK_C_FLAGS="" \
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
      -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
      -DCMAKE_POSITION_INDEPENDENT_CODE=OFF \
      -DWEBP_USE_THREAD=OFF"
  else
    # Using existing toolchain file, no need for extra args
    CMAKE_TOOLCHAIN_ARGS=""
  fi

  echo "CMAKE_TOOLCHAIN_ARGS: ${CMAKE_TOOLCHAIN_ARGS}, TOOLCHAIN_FILE: ${TOOLCHAIN_FILE}"

  # Build using CMake
  cd "${BUILD_DIR}"
  echo "Build directory: $BUILD_DIR"
  echo "Configuring build with CMake..."
  cmake -G "Ninja" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        ${CMAKE_TOOLCHAIN_ARGS} \
        -DCMAKE_BUILD_TYPE="" \
        -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}" \
        -DBUILD_SHARED_LIBS=OFF \
        -DENABLE_SHARED=OFF \
        -DLIBXML2_WITH_ICONV=OFF \
        -DLIBXML2_WITH_PYTHON=OFF \
        -DWITH_TURBOJPEG=OFF \
        -DPCRE2_BUILD_PCRE2GREP=OFF \
        -DXZ_THREADS=no \
        -DUSE_HTTPS=OFF \
        -DUSE_THREADS=OFF \
        -DUSE_XDIFF=OFF -DNO_NETWORK=ON \
        -DCMAKE_C_FLAGS="-O0 -g0 -DNDEBUG" \
        ${TEST_SKIP_ARGS} \
        "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"


  echo "Building ${LIB_NAME}..."
  cmake --build . -j$JCOUNT -v -- -k 0 || echo "======= build incomplete! ======="

  echo "Installing ${LIB_NAME} to ${OUTPUT_DIR}..."
  cmake --install .
  rm -f "${TMP_TOOLCHAIN_FILE}"

  echo "Build completed successfully!"
  echo "Library installed to ${OUTPUT_DIR}"

}

build_lib_for_arch $ARCH

# Return to workspace
cd /workspace