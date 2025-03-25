#!/bin/bash
# download-libs.sh - Check for library sources or download them

set -e

LIB_NAME=${1:-"zlib"}
LIB_VERSION=${2:-"1.3.1"}
SOURCES_DIR="/workspace/sources"

# Create sources directory if it doesn't exist
mkdir -p "${SOURCES_DIR}"
cd "${SOURCES_DIR}"

# Check if the library source already exists
if [ -d "${LIB_NAME}-${LIB_VERSION}" ]; then
  echo "Source directory ${LIB_NAME}-${LIB_VERSION} already exists, skipping download."
  exit 0
fi

echo "Source directory ${LIB_NAME}-${LIB_VERSION} not found."
echo "Please manually download and extract the source code to:"
echo "${SOURCES_DIR}/${LIB_NAME}-${LIB_VERSION}"
echo ""
echo "For zlib, you can download from:"
echo "  - https://github.com/madler/zlib/releases/download/v${LIB_VERSION}/zlib-${LIB_VERSION}.tar.gz"
echo "  - https://www.zlib.net/fossils/zlib-${LIB_VERSION}.tar.gz"
echo ""
echo "After downloading, extract with:"
echo "  tar xf zlib-${LIB_VERSION}.tar.gz -C ${SOURCES_DIR}"
echo ""
exit 1