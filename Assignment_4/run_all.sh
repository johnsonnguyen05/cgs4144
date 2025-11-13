#!/bin/bash
# Wrapper script to run all Assignment 4 analyses
# Uses python3 explicitly

cd "$(dirname "$0")/scripts"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# Run the master script
python3 run_all_analyses.py "$@"

