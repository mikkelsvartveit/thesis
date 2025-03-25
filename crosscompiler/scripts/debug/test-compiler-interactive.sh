arch=$1

echo "Running container for $arch"
docker run -it --entrypoint /bin/bash \
      -v "./sources:/workspace/sources" \
      -v "./output/${arch}:/workspace/output/${arch}" \
      -v "./patches:/workspace/patches" \
      -v "./toolchains:/workspace/toolchains" \
      -v "./scripts/build-lib.sh:/usr/local/bin/build-lib" \
      -v "./scripts/download-libs.sh:/usr/local/bin/download-libs" \
      "cross-compile-${arch}:latest"