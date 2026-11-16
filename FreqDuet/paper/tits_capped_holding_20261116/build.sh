#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3 summarize.py
mkdir -p output/pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf main.tex supplementary.tex
if rg -n 'LaTeX Warning:.*(undefined|multiply defined)|Citation .* undefined|Overfull \\[hv]box' output/pdf/{main,supplementary}.log; then
    exit 1
fi
printf 'PAPER_BUILD_OK\n'
