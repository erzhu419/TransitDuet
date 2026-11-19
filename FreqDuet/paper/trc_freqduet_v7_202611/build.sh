#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3 summarize_results.py
python3 generate_figures.py --stage full
out=${1:-build}
mkdir -p "$out"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$out" main.tex
if rg -ni 'warning.*(undefined|multiply defined)|there were undefined' "$out/main.log"; then
    exit 1
fi
