SIF_IMAGES="./singularity-images/crosscompiler-images/*.sif"
for SIF_IMAGE in $SIF_IMAGES; do
    arch=$(basename "${SIF_image}" .sif | sed 's/crosscompiler-//')

    echo "Processing architecture: ${arch}..."
    singularity exec $SIF_IMAGE bash -c '$TARGET-gcc -dumpmachine; \
        $TARGET-gcc --target-help < /dev/null | grep endian; \
        echo | $TARGET-gcc -E -dM - | grep ENDIAN; \
        $TARGET-gcc --help=target | grep endian; \
        $TARGET-gcc -dumpspecs | grep -i endian'
    echo ""
done

# c6x-unknown-uclinux
# csky-unknown-linux-gnu
# m32r-unknown-elf ❌
# mcore-unknown-elf ✅ (yes but not reflected in the dissassebly by objdump (transformed into big) but only endianness change is data, ie 1234 -> 3412)
# moxie-unknown-elf ✅
# nds32le-unknown-linux-gnu ✅
# nios2-unknown-linux-gnu
# rx-unknown-elf ✅ (yes but not reflected in the dissassebly by objdump (transformed into big), but only endianness change is data, ie 1234 -> 3412)
# tilegx-unknown-linux-gnu ✅