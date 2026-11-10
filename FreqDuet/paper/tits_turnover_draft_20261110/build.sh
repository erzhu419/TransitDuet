#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build_tables.py
latexmk -pdf -outdir=output/pdf -interaction=nonstopmode -halt-on-error main.tex
