#!/bin/bash

# Error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Function to clean up in case of errors
cleanup() {
    local exit_code=$?
    echo "Cleaning up temporary files..."
    # Remove zip file if it exists
    [ -f master.zip ] && rm master.zip
    # Remove extracted directory if it exists
    [ -d cpu_rec-master ] && rm -rf cpu_rec-master
    # If there was an error, remove the corpus folder too
    if [ $exit_code -ne 0 ]; then
        [ -d cpu_rec_corpus ] && rm -rf cpu_rec_corpus
        echo "An error occurred. All files have been cleaned up."
    fi
    exit $exit_code
}

# Set up trap for cleanup
trap cleanup EXIT

echo "Starting download of repository..."

# Download the repository
if ! wget -c https://github.com/airbus-seclab/cpu_rec/archive/refs/heads/master.zip; then
    echo "Failed to download repository"
    exit 1
fi

echo "Extracting files..."

# Extract the zip file
if ! unzip master.zip; then
    echo "Failed to extract zip file"
    exit 1
fi

echo "Moving cpu_rec_corpus folder..."

# Check if the corpus folder exists
if [ ! -d "cpu_rec-master/cpu_rec_corpus" ]; then
    echo "cpu_rec_corpus folder not found in the extracted files"
    exit 1
fi

# Move the corpus folder to current directory
if ! mv cpu_rec-master/cpu_rec_corpus .; then
    echo "Failed to move cpu_rec_corpus folder"
    exit 1
fi

echo "Unpacking xz files"

find cpu_rec_corpus -type f -name "*.xz" -print0 | while IFS= read -r -d '' file; do
    echo "Unpacking: $file"
    if ! xz -d "$file"; then
        echo "Failed to unpack: $file"
        exit 1
    fi
done

# Remove the downloaded zip and extracted directory
rm master.zip
rm -rf cpu_rec-master

echo "Successfully extracted cpu_rec_corpus folder"
echo "All temporary files have been cleaned up"