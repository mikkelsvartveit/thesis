#!/bin/bash
SCRIPT_DIR=$(pwd)
rm -rf results

find output -path "*install*" -name "*.a" | sort | while read file; do
    # Extract the architecture dir
    arch=$(echo "$file" | cut -d'/' -f2)
    
    # Get just the filename without the path
    filename=$(basename "$file")
    

    mkdir -p "results/library_files/$arch"
    mkdir -p "results/text_bin/$arch"
    mkdir -p "results/text_asm/$arch"
    
    echo "Processing: $file"
    
    cp "$file" "results/library_files/$arch/$filename"

    singularity exec \
      "$SCRIPT_DIR/slurm-scripts/cross-compiler-images/cross-compiler-${arch}.sif" \
      bash -c '${TARGET}-objcopy -S -j .text '"$file"' "results/text_bin/'$arch'/'$filename'.bin" && \
       echo "Sample elf header" > "results/text_asm/'$arch'/'$filename'.asm" && \
       ${TARGET}-readelf -h '"$file"' | head -n 40 >> "results/text_asm/'$arch'/'$filename'.asm" && \
       echo "\n ======================== \n" >> "results/text_asm/'$arch'/'$filename'.asm" && \
       ${TARGET}-objdump -d -j .text '"$file"' >> "results/text_asm/'$arch'/'$filename'.asm"' &
done