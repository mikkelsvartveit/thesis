import json
import os
from pathlib import Path
import statistics

import pandas as pd


def analyze_code_files(root_dir):
    # Walk through all directories
    dir_stats = []
    for dirpath, dirnames, filenames in os.walk(root_dir):

        for dir in sorted(dirnames):
            architecture_metadata_info(Path(dirpath) / dir, dir)

        # Filter for .code files
        code_files = [f for f in filenames if f.endswith(".code")]

        if code_files:  # Only process directory if it has .code files
            print(f"\nDirectory: {dirpath}")

            # Get file sizes
            file_sizes = []
            for dir in code_files:
                file_path = Path(dirpath) / dir
                size_bytes = file_path.stat().st_size
                file_sizes.append((dir, size_bytes))

            # Calculate statistics
            sizes_bytes = [size for _, size in file_sizes]

            # Find biggest and smallest files
            biggest_file = max(file_sizes, key=lambda x: x[1])
            smallest_file = min(file_sizes, key=lambda x: x[1])
            mean_size = statistics.mean(sizes_bytes)
            median_size = statistics.median(sizes_bytes)

            dir_stats.append(
                {
                    "directory": dirpath.split("/")[-1],
                    "file_sizes": file_sizes,
                    "biggest_file": biggest_file,
                    "smallest_file": smallest_file,
                    "mean_size": mean_size,
                    "median_size": median_size,
                    "total_files": len(code_files),
                }
            )

            # Print individual file sizes
            # print("\nFile sizes:")
            # for file, size in file_sizes:
            #     print(f"  {file}: {convert_size(size)}")

    # Sort file_stats by mean_size
    dir_stats.sort(key=lambda x: x["mean_size"])

    # Print sorted statistics
    print("\nDirectories sorted by mean file size (smallest to largest):")
    for dir in dir_stats:
        dirpath = dir["directory"]
        code_files = dir["file_sizes"]
        biggest_file = dir["biggest_file"]
        smallest_file = dir["smallest_file"]
        mean_size = dir["mean_size"]
        median_size = dir["median_size"]

        print(f"\nDirectory: {dirpath}")
        print("Statistics:")
        print(f"  Biggest file: {biggest_file[0]} ({convert_size(biggest_file[1])})")
        print(f"  Smallest file: {smallest_file[0]} ({convert_size(smallest_file[1])})")
        print(f"  Mean size: {convert_size(mean_size)}")
        print(f"  Median size: {convert_size(median_size)}")
        print(f"  Total files: {len(code_files)}")

    mean_sizes = [dir["mean_size"] for dir in dir_stats]
    print("\nOverall statistics:")
    print(f"Mean size: {convert_size(statistics.mean(mean_sizes))}")

    # count num files from each dir that is larger than 224^2
    print("\nNumber of files larger than 1024 bytes:")
    for dir in dir_stats:
        dirpath = dir["directory"]
        code_files = dir["file_sizes"]
        big_files = [f for f in code_files if f[1] >= 1024]
        print(f"  {dirpath}: {len(big_files)}")

    pandas_data = {
        "Directory": [dir["directory"] for dir in dir_stats],
        "Mean size (KB)": [f"{convert_kb(dir["mean_size"]):.2f}" for dir in dir_stats],
        "Median size (KB)": [
            f"{convert_kb(dir["median_size"]):.2f}" for dir in dir_stats
        ],
        "Files > 1024 bytes": [
            len([f for f in dir["file_sizes"] if f[1] >= 1024]) for dir in dir_stats
        ],
        "Files > 224^2 bytes": [
            len([f for f in dir["file_sizes"] if f[1] >= 224**2]) for dir in dir_stats
        ],
        "Total files": [dir["total_files"] for dir in dir_stats],
    }
    df = pd.DataFrame(pandas_data)

    path_to_save = Path(root_dir).parent / "filesizes.csv"
    print(f"\nSaving file sizes to {path_to_save}")
    df.to_csv(path_to_save, index=False)


def convert_size(size_bytes):
    # Convert bytes to human readable format
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def convert_kb(size_bytes):
    # Convert bytes to human readable format
    return size_bytes / 1024


def architecture_metadata_info(architecture_path, architecture_name):
    architecture_path = Path(architecture_path)
    # get file with achitecture_name.json
    architecture_path = architecture_path / f"{architecture_name}.json"
    with open(architecture_path, "r") as f:
        metadata = json.load(f)
        # get first as relevant metadata are the same for all files
        metadata0 = metadata[0]

    metadata0 = {
        key: metadata0[key] for key in ["architecture", "endianness", "wordsize"]
    }

    endianness = set()
    wordsize = set()
    for file in metadata:
        endianness.add(file["endianness"])
        wordsize.add(file["wordsize"])

    if len(endianness) > 1:
        raise ValueError(f"Multiple endianness found: {endianness}")
    if len(wordsize) > 1:
        raise ValueError(f"Multiple wordsize found: {wordsize}")

    print(metadata0)


# Usage
root_directory = "./ISAdetect/ISAdetect_full_dataset"  # Current directory, change this to your root directory
analyze_code_files(root_directory)
