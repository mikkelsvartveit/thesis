#!/bin/bash
set -e

force_rebuild=${1:-false}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

if [ ! -f "./singularity-images/crosscompiler-base.tar" ]; then
    if [ "$force_rebuild" = true ] || ! docker image inspect crosscompiler-base:latest &>/dev/null; then
        echo "Building crosscompiler-base Docker image..."
        docker build -t crosscompiler-base:latest -f "${PROJECT_ROOT}/singularity-images/Dockerfile.base" "${PROJECT_ROOT}"
        echo "Saving crosscompiler-base Docker image to tar file..."
        docker save -o ./singularity-images/crosscompiler-base.tar crosscompiler-base:latest
    else
        echo "Saving existing crosscompiler-base Docker image to tar file..."
        docker save -o ./singularity-images/crosscompiler-base.tar crosscompiler-base:latest
    fi
else
    echo "crosscompiler-base.tar file already exists, skipping build"
fi

if [ ! -f "./singularity-images/buildcross-base.tar" ]; then
    if [ "$force_rebuild" = true ] || ! docker image inspect buildcross-base:latest &>/dev/null; then
        echo "Building buildcross-base Docker image..."
        docker build -t buildcross-base:latest -f "${PROJECT_ROOT}/singularity-images/Dockerfile.buildcross-base" "${PROJECT_ROOT}"
        echo "Saving buildcross-base Docker image to tar file..."
        docker save -o ./singularity-images/buildcross-base.tar buildcross-base:latest
    else
        echo "Saving existing buildcross-base Docker image to tar file..."
        docker save -o ./singularity-images/buildcross-base.tar buildcross-base:latest
    fi
else
    echo "buildcross-base.tar file already exists, skipping build"
fi



# scp ./singularity-images/crosscompiler-base.tar idun:~/crosscompiler_v2/singularity-images/ 
# scp ./singularity-images/buildcross-base.tar idun:~/crosscompiler_v2/singularity-images/
