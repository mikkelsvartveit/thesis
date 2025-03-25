#!/bin/bash
set -e

force_rebuild=${1:-false}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

if [ ! -f "./base-images/cross-compile-base.tar" ]; then
    if [ "$force_rebuild" = true ] || ! docker image inspect cross-compile-base:latest &>/dev/null; then
        echo "Building cross-compile-base Docker image..."
        docker build -t cross-compile-base:latest -f "${PROJECT_ROOT}/Dockerfile.base" "${PROJECT_ROOT}"
        echo "Saving cross-compile-base Docker image to tar file..."
        docker save -o ./base-images/cross-compile-base.tar cross-compile-base:latest
    else
        echo "Saving existing cross-compile-base Docker image to tar file..."
        docker save -o ./base-images/cross-compile-base.tar cross-compile-base:latest
    fi
else
    echo "cross-compile-base.tar file already exists, skipping build"
fi

if [ ! -f "./base-images/buildcross-base.tar" ]; then
    if [ "$force_rebuild" = true ] || ! docker image inspect buildcross-base:latest &>/dev/null; then
        echo "Building buildcross-base Docker image..."
        docker build -t buildcross-base:latest -f "${PROJECT_ROOT}/Dockerfile.buildcross-base" "${PROJECT_ROOT}"
        echo "Saving buildcross-base Docker image to tar file..."
        docker save -o ./base-images/buildcross-base.tar buildcross-base:latest
    else
        echo "Saving existing buildcross-base Docker image to tar file..."
        docker save -o ./base-images/buildcross-base.tar buildcross-base:latest
    fi
else
    echo "buildcross-base.tar file already exists, skipping build"
fi



# scp ./base-images/cross-compile-base.tar idun:~/crosscompiler_v2/base-images/ 
# scp ./base-images/buildcross-base.tar idun:~/crosscompiler_v2/base-images/
