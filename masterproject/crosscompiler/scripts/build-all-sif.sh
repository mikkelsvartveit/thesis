#!/bin/bash
# build-all.sh - Build libraries for all architectures

set -e

# Architectures to build for
SIF_IMAGES="./singularity-images/crosscompiler-images/*.sif"
ONLY_ARCH=${1:-}

# Libraries to build
LIBRARIES=(
  "zlib:1.3"
  "libjpeg-turbo:3.1.0"
  "libpng:1.6.47"
  "freetype:2.13.3"
  "xzutils:1"
  "pcre2:10.45"
  "libyaml:0.2.5"
  "libwebp:1.5.0"
  # "libxml2:2.14"
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
  if [ -n "${ONLY_ARCH}" ] && [ "${arch}" != "${ONLY_ARCH}" ]; then
    continue
  fi
  
  echo "Processing architecture: ${arch}..."
  
  # Create output directory
  mkdir -p "${PROJECT_ROOT}/output/${arch}"
  
  # Build all libraries for this architecture using a one-liner for loop
  echo "Building all libraries for ${arch}..."
  
  # Convert LIBRARIES array to a Bash-compatible string
  LIB_ARRAY_STR=$(printf "'%s' " "${LIBRARIES[@]}")
  (
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
        rm -rf /workspace/output/${arch}/\${lib_name}; \
        build-lib \"\${lib_name}\" \"\${lib_version}\" || echo \"====== Failed to build ${lib_name} ======\"; \
      done" > "${PROJECT_ROOT}/output/${arch}/build.log" 2>&1 
    echo "Done building ${arch}"
  ) &
done
wait

# Print summary
echo -e "\nBuild Summary:"
for SIF_image in $SIF_IMAGES; do
    temp_res=$(mktemp)
    # Extract architecture name from SIF image filename
    arch=$(basename "${SIF_image}" .sif | sed 's/cross-compiler-//')
    if [ -z "${ONLY_ARCH}" ] || [ "${arch}" == "${ONLY_ARCH}" ]; then
      echo "Architecture: ${arch}" >> "${temp_res}"
      TOTAL_SIZE=0
      # Convert LIBRARIES array to iterate through each library
      for lib_info in "${LIBRARIES[@]}"; do
        IFS=':' read -r lib_name lib_version <<< "${lib_info}"
        lib_dir="${PROJECT_ROOT}/output/${arch}/${lib_name}/install/lib"
        # Check if any *.a file exists in the directory
        if [ -n "$(find "${lib_dir}" -name "*.a" -print -quit 2>/dev/null)" ]; then
          # Find the largest static library file
          main_lib=$(find "${lib_dir}" -name "*.a" -type f -exec ls -s {} \; | sort -nr | head -1 | awk '{print $2}')
          
          tmp_file=$(mktemp)
          # Use singularity to examine the .text section size
          text_size=$(singularity exec "${SIF_image}" \
                bash -c '${TARGET}-objcopy -S -j .text '"${main_lib}"' '"${tmp_file}"' && wc -c < '"${tmp_file}")
          rm -f "${tmp_file}"
          TOTAL_SIZE=$((TOTAL_SIZE + text_size))
          
          if [ "$text_size" != "N/A" ] && [ -n "$text_size" ]; then
              # Convert to human-readable format locally
              text_size_human=$(echo $text_size | awk '{printf "%.2f KB", $1/1024}')
              echo "  - ${lib_name} ${lib_version}: SUCCESS (.text: ${text_size_human})" >> "${temp_res}"
          else
              echo "  - ${lib_name} ${lib_version}: SUCCESS (.text: size unavailable)" >> "${temp_res}"
          fi
        else
            echo "  - ${lib_name} ${lib_version}: FAILED (no static library found)" >> "${temp_res}"
        fi
      done
      TOTAL_SIZE_HUMAN=$(singularity exec "${SIF_image}" bash -c "echo $TOTAL_SIZE | awk '{printf \"%.2f KB\", \$1/1024}'")
      echo "Estimated .text section size: ${TOTAL_SIZE_HUMAN}" >> "${temp_res}"
      cat "${temp_res}"
    fi
    rm -f "${temp_res}"
done
wait
echo "done"