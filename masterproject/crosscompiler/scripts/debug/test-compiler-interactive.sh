arch=$1

echo "Running container for $arch"
docker run -it --entrypoint /bin/bash \
      -v "./sources:/workspace/sources" \
      -v "./output:/workspace/output" \
      -v "./patches:/workspace/patches" \
      -v "./toolchains:/workspace/toolchains" \
      -v "./scripts:/workspace/scripts" \
      "cross-compiler-${arch}:latest"