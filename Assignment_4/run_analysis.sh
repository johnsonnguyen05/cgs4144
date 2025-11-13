#!/bin/bash
# Master script to run all analyses with venv activation
# Usage: ./run_analysis.sh [options]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate venv
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "Error: Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please create a venv or install packages in your Python environment"
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, scipy, statsmodels" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing packages..."
    pip install -q pandas numpy scikit-learn matplotlib seaborn scipy statsmodels
fi

# Check requirements
echo ""
echo "Checking file requirements..."
python "$SCRIPT_DIR/scripts/check_requirements.py"
echo ""

# Change to scripts directory and run
cd "$SCRIPT_DIR/scripts"
python run_all_analyses.py "$@"

