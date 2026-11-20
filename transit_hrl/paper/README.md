# Plan-validity replanning: review PDF and analysis artifact

[Review PDF](submission.pdf) | [Markdown source](manuscript.md) |
[Evidence map](evidence_map.md) | [Readiness summary](submission_readiness.md) | [Bundle manifest](artifact_files.txt)

The PDF is a venue-neutral, author-neutral draft. Its title, text, tables,
captions and citations come from the Markdown and BibTeX sources, embedding
five vector PDF figures. For the official-style, double-blind TMLR PDF and
supplementary ZIP, see the [TMLR build and submission instructions](tmlr/README.md).

## Rebuild

From `transit_hrl/` or the root of the extracted bundle, with Python 3.10+,
GNU Make, and Poppler (`pdftotext`). The reference PDF build uses
**Pandoc 3.9, latexmk 4.76 and pdfTeX 1.40.22 (TeX Live 2022/dev/Debian)**.
Install a full working LaTeX format and the fonts and packages used by the
templates; on Debian the build uses `latexmk`, `texlive-latex-extra`,
`texlive-fonts-recommended` and `lmodern`. Verify that
`kpsewhich -engine=pdftex -format=fmt pdflatex.fmt` finds the format before
building. Pandoc 3.1.11.1 does not support the current Lua filter.

```sh
python3 -m pip install -r paper/requirements.txt
make -C paper reproduce
```

If Pandoc is not on `PATH`, pass `PANDOC=/absolute/path/to/pandoc`.
`make -C paper review-bundle` writes `/tmp/freqhrl-review.tar.gz`. This
review package excludes scheduler scripts, cluster paths, job receipts,
and archived operational notes. The full local working package is built
with `make -C paper bundle` (`/tmp/freqhrl-submission.tar.gz`).

## Run the frozen PointMaze example

From the same directory, with Python 3.10 on Linux:

```sh
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python3 -m pip install -r paper/requirements-replay.txt
make -C paper replay
```

This runs the original trigger and controller code without training, a GPU,
rendering or a scheduler. The example is the first root and first held path
of the PointMaze yoked-clock study (root `1981400000`, path `1987400100`).
It verifies all 24 call times against the [replay fixture](data/pointmaze_replay.json)
and compares the 1,200-step episode ISE against the saved evaluation
(`1.0109450548372947`, absolute tolerance `1e-5`). The expected call times
were recorded from the shipped replay runtime; the reference ISE comes from
the original study. This is a single-episode check, not a new efficacy test.

## Data and scope

| Outputs | Saved inputs | What the rebuild does |
| --- | --- | --- |
| Table 1, Figures 1 and 4 | `paper/data/stage9/replicate_*.json`, eight optimizer roots | Recompute paired root effects and Student-t intervals; use saved call times for Figure 1 |
| Table 8 | PointMaze timing portfolio summary | Check means, contrasts and intervals against the frozen follow-up |
| Table 2, Figure 2a | Label-free query ablation summary (`B = 24`) | Recompute paired eight-root contrasts for clock-grid and observable-change query acquisition |
| Table 3, Figure 5a | PointMaze timing baselines and MLP summary | Verify eight-root contrasts against native periodic, learned binary, distance threshold, cost difference, and raw MLP |
| Path-specific timing and supervision | PointMaze matched review-controls summary | Recompute root-level ISE intervals for the yoked clock and matched-step single-arm supervision |
| Tables 4 and 9, Figures 2b-d and 5c,d | Multi-budget and CEM summaries (`B = 12..40`) | Verify same- and cross-budget ISE against periodic and distance timing, plus planner, scoring and measured control time |
| Table 5, Figure 5b | `Reacher-v5` polygonal 8-seed evaluation summary | Verify unfiltered root contrasts, Holm sensitivity and waypoint-authority strata |
| Discussion | Local-episode and matched-target development summaries | Recompute the local versus episode gains and the waiting-target comparison on two roots per task |
| Table 10 | Smooth-ellipse Reacher and LQR cart-pole applicability summaries | Check paired applicability contrasts and preserve both diagnostic qualification outcomes |
| Tables 6-7, Figure 3 | Qualified Stage298/301/304 Transit reports and event audit | Preserve single-intervention contrasts and the negative two-decision episode boundary |

The results rebuild uses saved outcomes without running a simulator or scheduler.
The single replay checkpoint does not reproduce policy training or all evaluation cohorts.
