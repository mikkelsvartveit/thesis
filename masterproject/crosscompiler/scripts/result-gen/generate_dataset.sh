#!/bin/bash
SCRIPT_DIR=$(pwd)
set -e

# Create main results directories
rm -rf results
mkdir -p results/library_files
mkdir -p results/text_bin
mkdir -p results/text_asm

lib_filters=(
    libpng.a
    liblibpng16_static.a
    libpcre2-posix.a
    libzlibstatic.a
)

# First loop through all architecture directories
for arch_dir in output/*/; do
    # Extract just the architecture name
    arch=$(basename "$arch_dir")
    echo "arch_dir: $arch_dir"
        
    # Create architecture-specific result directories
    mkdir -p "results/library_files/$arch"
    mkdir -p "results/text_bin/$arch"
    mkdir -p "results/text_asm/$arch"
    
    # Find all .a files in install directories for this architecture
    find "$arch_dir" -path "*install*" -name "*.a" | sort | while read file; do
        # Get just the filename without the path
        filename=$(basename "$file")

        skip=0
        for excluded in "${lib_filters[@]}"; do
            if [[ "$filename" == "$excluded" ]]; then
                echo "  $arch: skipping excluded library $filename"
                skip=1
                break
            fi
        done
        if [ $skip -eq 1 ]; then
            continue
        fi
        
        echo "  $arch: processing library $filename, file: $file"
        
        # Copy the library file with full path
        cp "$file" "results/library_files/$arch/$filename"
        
        # Pass the COPIED file to the extract script, not the original
        singularity exec \
          "$SCRIPT_DIR/slurm-scripts/cross-compiler-images/cross-compiler-${arch}.sif" \
          bash ./scripts/result-gen/extract_library.sh "$SCRIPT_DIR/results/library_files/$arch/$filename" "$arch" "$filename" &
    done
done

# Wait for all background processes to complete
wait
echo ""
echo "===== All processing complete ====="
echo ""


instruction_width_map() {
    local arch=$1
    
    local width=""
    local width_type=""
    local comment=""
    
    case "$arch" in
        # x86 architectures - variable instruction width
        "bfin")
            width="16/32"
            width_type="mixed"
            comment="Blackfin"
            ;;
        "c6x")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "cr16")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "cris")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "csky")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "epiphany")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "fr30")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "frv")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "ft32")
            width="32"
            width_type="fixed"
            comment="disassembled in both little and big, even though compiled for little"
            ;;
        "h8300")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "iq2000")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "kvx")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "lm32")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "m32r")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "m68k-elf")
            width="16/32/48"
            width_type="mixed"
            comment=""
            ;;
        "m68k-uclibc")
            width="16/32/48"
            width_type="mixed"
            comment=""
            ;;
        "mcore")
            width="16"
            width_type="fixed"
            comment="Says 32 in elf header, but instructions are 16-bit? investigate"
            ;;
        "mmix")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "mn10300")
            width="na"
            width_type="variable"
            comment=""
            ;;
        "moxie")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "msp430")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "nds32")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "pdp11")
            width=""
            width_type=""
            comment=""
            ;;
        "rl78")
            width="na"
            width_type="variable"
            comment=""
            ;;
        "rx")
            width="na"
            width_type="variable"
            comment=""
            ;;
        "tilegx")
            width="64"
            width_type="fixed"
            comment="Mix of VLIW and variable width, but instructions are fixed 64-bit apart. sometimes two instructions are packed into one 64-bit word"
            ;;
        "tricore")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "v850")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "visium")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "xstormy16")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "xtensa")
            width="na"
            width_type="variable"
            comment="either 16 or 24 bits, does not align to a single boundary"
            ;;
        *)
            echo "Unknown architecture: $arch"
            exit 1
            ;;
    esac

    echo "$width;$width_type;$comment;"
}

analyze_elf() {
    local binary=$1
    local arch_name=$2
        
    # Use readelf to get ELF header information
    local header=$(readelf -h "$binary")
    
    # Extract endianness
    if echo "$header" | grep -q "little endian"; then
        endianness="little"
    elif echo "$header" | grep -q "big endian"; then
        endianness="big"
    else
        endianness="unknown"
    fi
    
    # Extract word size (32-bit or 64-bit)
    if echo "$header" | grep -q "ELF32"; then
        wordsize="32"
    elif echo "$header" | grep -q "ELF64"; then
        wordsize="64"
    else
        wordsize="unknown"
    fi
    echo "$endianness;$wordsize;"
}


csv_report="results/report.csv"
echo "Architecture,Libraries,Text_Size" > "$csv_report"
csv_labels="results/labels.csv"
echo "architecture;endianness;wordsize;instructionwidth_type;instructionwidth;comment" > $csv_labels

report_file="results/report.txt"
rm -f $report_file
touch $report_file

echo "Waiting for file system to update"
for i in {10..1}; do
    echo -ne "\r${i}s  "
    sleep 1
done
echo -e "\rWait complete!          "
# Loop through all architecture directories
for arch_dir in results/text_bin/*/; do
    # Extract just the architecture name
    arch=$(basename "$arch_dir")
    
    echo "Processing report for $arch"
    echo "Architecture: $arch" >> "$report_file"
    
    # if arch dir is empty
    if [ ! "$(ls -A $arch_dir)" ]; then
        echo "No libraries found for $arch"
        echo "No libraries found" >> "$report_file"
        echo "" >> "$report_file"
        echo ""
        continue
    fi

    # Loop through all library files
    TOTAL_SIZE=0
    for file in "$arch_dir"*.bin; do

        # Get just the filename without the path
        filename=$(basename "$file" .bin)
        
        # Get the text size from the corresponding .asm file
        text_size=$(wc -c "$file" | awk '{print $1}')
        TOTAL_SIZE=$((TOTAL_SIZE+text_size))
        kb_size=$(echo $text_size | awk '{print $1/1024 " KB"}')
        
        # Write the results to the CSV file
        echo "$arch,$filename,$kb_size" >> "$csv_report"
        
        # Write the results to the text file
        echo "  Library: $filename, Text Size $kb_size" >> "$report_file"
    done
    echo "  Total text size: $((TOTAL_SIZE/1024)) KB" >> "$report_file"
    echo "" >> "$report_file"
    echo ""

    # Extract endianness and wordsize
    arch_name=$(echo $arch | sed 's/-.*//')
    elf_file=$(ls results/library_files/$arch/*.a | head -1)
    endian_wordsize=$(analyze_elf $elf_file $arch_name)
    instr_w=$(instruction_width_map $arch_name)
    echo "$arch;$endian_wordsize$instr_w" >> $csv_labels
done


