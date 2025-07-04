#!/bin/bash

# Default configuration values
NODES=1
CPUS=8
MEM="16G"
TIME="01:00:00"
PARTITION="GPUQ"

# Get script name for usage message
SCRIPT_NAME=$(basename "$0")

# Help function
usage() {
    echo "Usage: $SCRIPT_NAME [-t time] [-n nodes] [-c cpus] [-m memory] [-p partition]"
    echo "Options:"
    echo "  -t: Wall time (default: $TIME)"
    echo "  -a: Account name (default: $ACCOUNT)"
    echo "  -n: Number of nodes (default: $NODES)"
    echo "  -c: CPUs per task (default: $CPUS)"
    echo "  -m: Memory (default: $MEM)"
    echo "  -p: Partition (default: $PARTITION)"
    echo "  -h: Show this help message"
    exit 1
}

# Parse command line options
while getopts "t:a:n:c:m:p:h" opt; do
    case $opt in
        t)  if [[ $OPTARG =~ ^[0-9]+$ ]]; then
                # If just a number is provided, treat it as hours
                TIME="$OPTARG:00:00"
            else
                # Otherwise use the provided time format
                TIME="$OPTARG"
            fi
            ;;
        n)  NODES="$OPTARG"
            ;;
        c)  CPUS="$OPTARG"
            ;;
        m)  MEM="$OPTARG"
            ;;
        p)  PARTITION="$OPTARG"
            ;;
        h)  usage
            ;;
        ?)  usage
            ;;
    esac
done

# Launch salloc with specified parameters
salloc \
    --time="$TIME" \
    --nodes="$NODES" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --partition="$PARTITION" \
    --gres="gpu:a100:1"
