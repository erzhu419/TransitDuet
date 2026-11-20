#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3 render_evidence.py
out=build
if [[ $# -gt 0 ]]; then out=$1; fi
mkdir -p "$out"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$out" main.tex
if rg -ni 'warning.*(undefined|multiply defined)|there were undefined' "$out/main.log"; then
    exit 1
fi
