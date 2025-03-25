file=$1
arch=$2
filename=$3

# Create a temporary directory for object extraction
TEMP_DIR=$(mktemp -d)

# Extract all objects from the archive
# Using cd to change directory before extraction since --output isn't supported
cd $TEMP_DIR
if ! ${TARGET}-ar x "$file"; then
    echo "Error extracting $file. Check if file exists and is valid."
    exit 1
fi
cd - > /dev/null

# Create empty files for combined output
touch "$TEMP_DIR/combined.bin"
echo "Sample elf header" > "results/text_asm/$arch/$filename.asm"
${TARGET}-readelf -h "$file" | head -n 40 >> "results/text_asm/$arch/$filename.asm"
echo -e "\n ======================== \n" >> "results/text_asm/$arch/$filename.asm"

# Process each object file (adjust the wildcard to match actual extracted files)
for obj in $TEMP_DIR/*.o $TEMP_DIR/*.obj; do
    if [ -f "$obj" ]; then
        # Get just the object filename
        objname=$(basename "$obj")

        # Extract .text section to binary
        ${TARGET}-objcopy -O binary --only-section=.text "$obj" "$TEMP_DIR/${objname%.o}.bin" 2>/dev/null

        # Concatenate to the combined binary
        if [ -f "$TEMP_DIR/${objname%.o}.bin" ]; then
            cat "$TEMP_DIR/${objname%.o}.bin" >> "$TEMP_DIR/combined.bin"
        fi

        # Add disassembly to the combined asm file
        echo -e "\n ======== OBJECT: $objname ======== \n" >> "results/text_asm/$arch/$filename.asm"
        ${TARGET}-objdump -d -j .text "$obj" 2>/dev/null >> "results/text_asm/$arch/$filename.asm"
    fi
done

# Copy the combined binary to the results directory
cp "$TEMP_DIR/combined.bin" "results/text_bin/$arch/$filename.bin"

# Clean up
rm -rf "$TEMP_DIR"