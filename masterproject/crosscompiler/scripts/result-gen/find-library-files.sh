#!/bin/bash
SCRIPT_DIR=$(pwd)
set -e

# Create main results directories
rm -rf results
mkdir -p results/library_files
mkdir -p results/text_bin
mkdir -p results/text_asm

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


csv_report="results/report.csv"
echo "Architecture,Libraries,Text_Size" > "$csv_report"

report_file="results/report.txt"
rm -f $report_file
touch $report_file

echo "Waiting for fs to update"
sleep 1

# Loop through all architecture directories
for arch_dir in results/text_bin/*/; do
    # Extract just the architecture name
    arch=$(basename "$arch_dir")
    
    echo "Processing report for $arch"
    
    # if arch dir is empty
    if [ ! "$(ls -A $arch_dir)" ]; then
        echo "No libraries found for $arch"
        echo "Architecture: $arch" >> "$report_file"
        echo "No libraries found" >> "$report_file"
        echo "" >> "$report_file"
        echo ""
        continue
    fi

    # Loop through all library files
    echo "Architecture: $arch" >> "$report_file"
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
done