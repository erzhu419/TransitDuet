#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build_tables.py
python3 publication_tables.py
latexmk -pdf -outdir=output/pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -outdir=output/pdf -interaction=nonstopmode -halt-on-error supplementary.tex
python3 verify.py
