rm -rf cpu_rec
echo "Downloading CPU Rec dataset..."
wget -q --show-progress https://github.com/airbus-seclab/cpu_rec/archive/refs/heads/master.zip
echo "Unzipping..."
unzip -q master.zip
rm master.zip
mkdir cpu_rec
mv cpu_rec-master/cpu_rec_corpus cpu_rec/cpu_rec_corpus
rm -rf cpu_rec-master

cd cpu_rec/cpu_rec_corpus

echo "Decompressing files..."
for file in *.xz; do
    xz -d -q $file
done

echo "Done downloading dataset. Removing files:"
# rm files that starts with _
rm -v _*

# Array of filenames to remove
files_to_remove=(
    "CUDA.corpus"  # Probably NVIDIA PTX bytecode
    "OCaml.corpus"  # OCaml bytecode
    "WASM.corpus"  # WebAssembly bytecode
    # Special variants that aren't pure ISA
    "#6502#cc65.corpus"  # Compiler-specific variant
)

# Remove files in the array
for file in "${files_to_remove[@]}"; do
    rm -v "$file"
done

echo "Dataset ready"