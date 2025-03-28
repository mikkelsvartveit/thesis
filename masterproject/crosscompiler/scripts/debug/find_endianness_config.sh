SIF_IMAGES="./slurm-scripts/cross-compiler-images/*.sif"
for SIF_IMAGE in $SIF_IMAGES; do
    arch=$(basename "${SIF_image}" .sif | sed 's/cross-compiler-//')

    echo "Processing architecture: ${arch}..."
    singularity exec $SIF_IMAGE bash -c '$TARGET-gcc -dumpmachine; \
        $TARGET-gcc --target=$TARGET -E -dM - < /dev/null | grep ENDIAN; \
        echo | $TARGET-gcc -E -dM - | grep ENDIAN; \
        $TARGET-gcc --help=target | grep endian; \
        $TARGET-gcc -dumpspecs | grep -i endian'
    echo ""
done