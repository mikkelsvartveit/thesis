#!/usr/bin/env python3

import os
import sys
import shutil
import zipfile
import tarfile
import tempfile
from pathlib import Path
from typing import Union, List

# Set the fixed destination directory
DEST_DIR = Path("./test/dest")

def get_basename(filepath: Union[str, Path]) -> str:
    """Get the base name without .tar.gz extension."""
    return Path(filepath).stem.replace('.tar', '')

def is_dir_empty(directory: Union[str, Path]) -> bool:
    """Check if a directory is empty."""
    return not any(Path(directory).iterdir())

def confirm(prompt: str) -> bool:
    """Prompt for yes/no confirmation."""
    while True:
        response = input(f"{prompt} [y/n]: ").lower()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please answer y or n.")

def extract_targz(source_file: Path, dest_dir: Path) -> None:
    """Extract a tar.gz file to destination directory."""
    try:
        with tarfile.open(source_file, 'r:gz') as tar:
            tar.extractall(path=dest_dir)
    except Exception as e:
        raise Exception(f"Failed to extract {source_file}: {str(e)}")

def find_files_by_extension(directory: Path, extension: str) -> List[Path]:
    """Find all files with given extension in directory."""
    return list(directory.rglob(f"*.{extension}"))

def setup_destination(dest_dir: Path) -> None:
    """Setup destination directory, prompting for cleanup if needed."""
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    elif not is_dir_empty(dest_dir):
        if not confirm(f"Destination directory {dest_dir} is not empty. Delete existing contents?"):
            print("Operation cancelled by user")
            sys.exit(0)
        # Remove existing contents
        for item in dest_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

def process_files(zip_file: Path, dest_dir: Path) -> None:
    """Process the zip file and its contents."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract zip file
        print(f"Extracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
        except Exception as e:
            raise Exception(f"Failed to extract zip file: {str(e)}")

        # Move CSV files to destination
        print("Moving CSV files to destination directory...")
        for csv_file in find_files_by_extension(temp_path, "csv"):
            shutil.copy2(csv_file, dest_dir)

        # Process tar.gz files
        for targz_file in find_files_by_extension(temp_path, "tar.gz"):
            subfolder_name = get_basename(targz_file)
            subfolder_path = dest_dir / subfolder_name
            subfolder_path.mkdir(exist_ok=True)
            
            print(f"Extracting {targz_file} to {subfolder_path}...")
            extract_targz(targz_file, subfolder_path)

def main():
    """Main function."""
    # Check arguments
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <zip file>")
        print(f"Example: {sys.argv[0]} myfile.zip")
        sys.exit(1)

    zip_file = Path(sys.argv[1])

    # Validate input file
    if not zip_file.exists():
        print(f"Error: File '{zip_file}' not found")
        sys.exit(1)
    if zip_file.suffix.lower() != '.zip':
        print("Error: File must be a .zip file")
        sys.exit(1)

    try:
        # Setup destination directory
        setup_destination(DEST_DIR)

        # Process the files
        process_files(zip_file, DEST_DIR)

        print("All files have been processed successfully")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()