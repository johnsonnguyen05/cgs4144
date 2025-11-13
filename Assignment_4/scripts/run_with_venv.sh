#!/bin/bash
# Script to run Python scripts with venv automatically activated
# Usage: ./run_with_venv.sh script_name.py [args...]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Activate venv
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "Error: Virtual environment not found at $PROJECT_ROOT/venv"
    exit 1
fi

# Check if pandas is available
python -c "import pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: pandas not found in venv. Installing requirements..."
    pip install -q pandas numpy scikit-learn matplotlib seaborn scipy statsmodels
fi

# Change to scripts directory
cd "$SCRIPT_DIR"

# Run the script with all arguments
python "$@"

