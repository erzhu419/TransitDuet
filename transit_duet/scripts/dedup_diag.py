#!/usr/bin/env python3
"""
dedup_diag.py
=============
Deduplicate episode rows in diagnostics.csv files produced by resumed
TransitDuet runs (runner_v2.py appends rows on resume, creating dups
for any ep in [last_checkpoint+1 .. pre-timeout_ep]).

Keeps the LAST row per `ep` (most recent training trajectory).
Creates a `.predup.csv` backup next to any CSV that had duplicates.

Usage:
  python scripts/dedup_diag.py logs/A_full_seed123/diagnostics.csv
  python scripts/dedup_diag.py logs/                      # recurse
  python scripts/dedup_diag.py logs/ --dry-run            # report only
"""

import argparse
from pathlib import Path
import pandas as pd


EVAL_EP_MARKER = 9000  # rows with ep >= this are Pareto/eval rows; do not dedup


def dedup_csv(path: Path, dry_run: bool = False) -> int:
    df = pd.read_csv(path)
    if 'ep' not in df.columns:
        return 0
    n_before = len(df)
    train = df[df['ep'] < EVAL_EP_MARKER]
    evals = df[df['ep'] >= EVAL_EP_MARKER]
    train = (train.drop_duplicates(subset='ep', keep='last')
                  .sort_values('ep')
                  .reset_index(drop=True))
    df = pd.concat([train, evals], ignore_index=True)
    removed = n_before - len(df)
    if removed == 0 or dry_run:
        return removed
    bak = path.with_name(path.stem + '.predup.csv')
    if not bak.exists():
        path.rename(bak)
    else:
        path.unlink()
    df.to_csv(path, index=False)
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+',
                    help='CSV files or directories (recursed for diagnostics.csv)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report duplicates but do not rewrite files')
    args = ap.parse_args()

    targets = []
    for p in args.paths:
        pp = Path(p)
        if pp.is_dir():
            targets.extend(sorted(pp.rglob('diagnostics.csv')))
        elif pp.is_file():
            targets.append(pp)
        else:
            print(f"  [skip] {p}: not a file or dir")

    total = 0
    touched = 0
    for t in targets:
        removed = dedup_csv(t, dry_run=args.dry_run)
        if removed > 0:
            verb = 'would remove' if args.dry_run else 'removed'
            print(f"  {t}: {verb} {removed} duplicate row(s)")
            total += removed
            touched += 1
    suffix = ' (dry run)' if args.dry_run else ''
    print(f"Files affected: {touched}/{len(targets)}  "
          f"total duplicates: {total}{suffix}")


if __name__ == '__main__':
    main()
