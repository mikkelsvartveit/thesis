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
          "$SCRIPT_DIR/singularity-images/crosscompiler-images/crosscompiler-${arch}.sif" \
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
        "arc" | "arceb")
            width="16/32"
            width_type="mixed"
            comment="16 or 32, but sometimes 64 due to 32bit intermediate as arg. Disassembly same for both endians"
            ;;
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
        "loongarch64")
            width="64"
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
        "mcore" | "mcoreeb")
            width="16"
            width_type="fixed"
            comment="16 bit instructions, 32 bit register file https://www.nxp.com/docs/en/data-sheet/MMC2001RM.pdf"
            ;;
        "microblaze" | "microblazeel")
            width="64"
            width_type="fixed"
            comment=""
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
        "moxie" | "moxieel")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "msp430")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "nds32" | "nds32be")
            width="16/32"
            width_type="mixed"
            comment=""
            ;;
        "nios2")
            width="64"
            width_type="fixed"
            comment=""
            ;;
        "or1k")
            width="64"
            width_type="fixed"
            comment=""
            ;;
        "pdp11")
            width=""
            width_type=""
            comment=""
            ;;
        "pru")
            width="32"
            width_type="fixed"
            comment=""
            ;;
        "rl78")
            width="na"
            width_type="variable"
            comment=""
            ;;
        "rx" | "rxeb")
            width="na"
            width_type="variable"
            comment=""
            ;;
        "tilegx" | "tilegxbe")
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
    esac

    echo "$width;$width_type;$comment"
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
    elf_file=$(ls results/library_files/$arch/*.a | head -1)
    endian_wordsize=$(analyze_elf $elf_file $arch)
    instr_w=$(instruction_width_map $arch)
    echo "$arch;$endian_wordsize$instr_w" >> $csv_labels
done

echo "Report generated in $report_file"
echo "CSV report generated in $csv_report"
echo "CSV labels generated in $csv_labels"

echo "Compressing dataset results"
cd results
tar --sort=name \
    --owner=0 --group=0 --numeric-owner \
    --mtime='1970-01-01 00:00Z' \
    --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime \
    -c text_bin/ | gzip -n > text_bin.tar.gz
cd ..



