file=$1
arch=$2
filename=$3

get_code_section_headers() {
    local arch_name=$1
    local sections=""

    case $arch_name in
        "rx"|"rxeb")
            sections="P"
            ;;
        "ft32")
            sections=".text*"
            ;;
        *)
            sections=".text .text.*"
            ;;
    esac
    
    echo "$sections"
}

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

        # Get code sections for this architecture
        code_sections=$(get_code_section_headers "$arch")
        
        # Create a temporary file for this object's binary
        obj_bin="$TEMP_DIR/${objname%.o}.bin"
        touch "$obj_bin"
        
        # Extract each code section to the same binary file
        for section in $code_sections; do
            temp_section_bin="$TEMP_DIR/$objname-$section.bin"
            
            # Extract section to temporary file
            if ${TARGET}-objcopy -O binary --only-section="$section" "$obj" "$temp_section_bin" 2>/dev/null; then
                if [ -s "$temp_section_bin" ]; then
                    cat "$temp_section_bin" >> "$obj_bin"
                fi
            fi

            if ${TARGET}-objcopy -S --only-section="$section" "$obj" "$temp_section_bin" 2>/dev/null; then
                if [ -s "$temp_section_bin" ]; then
                    ${TARGET}-objdump -d "$temp_section_bin" >> "results/text_asm/$arch/$filename.asm"
                fi
            fi
            
            # Clean up temporary section file
            rm -f "$temp_section_bin"
        done

        # Concatenate to the combined binary
        if [ -s "$obj_bin" ]; then
            cat "$obj_bin" >> "$TEMP_DIR/combined.bin"
        fi
    fi
done

# Copy the combined binary to the results directory
cp "$TEMP_DIR/combined.bin" "results/text_bin/$arch/$filename.bin"

# Clean up
rm -rf "$TEMP_DIR"
