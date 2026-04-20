#!/bin/bash
# TransitDuet — one-line server launcher
# Usage:  bash run_all.sh                   # run everything
#         bash run_all.sh --quick           # 50-ep sanity check
#         bash run_all.sh --dry-run         # preview plan
#         bash run_all.sh --workers 4       # manual worker count

set -e
cd "$(dirname "$0")"

echo "==============================================="
echo "  TransitDuet Experiment Runner"
echo "  Started: $(date)"
echo "  Host: $(hostname)"
echo "==============================================="

# Basic requirements check
python3 -c "import torch, numpy, pandas, yaml, pygame" 2>/dev/null || {
    echo "ERROR: missing Python deps. Run: pip install -r requirements.txt"
    exit 1
}

# Run the launcher (auto-detects CPU/GPU)
exec python3 -u scripts/launcher.py "$@"
