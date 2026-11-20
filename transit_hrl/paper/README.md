# Plan-validity replanning: review PDF and analysis artifact

[Review PDF](submission.pdf) | [Markdown source](manuscript.md) |
[Evidence map](evidence_map.md) | [Readiness summary](submission_readiness.md) | [Bundle manifest](artifact_files.txt)

The PDF is a venue-neutral, author-neutral draft. Its title, text, tables,
captions and citations come from the Markdown and BibTeX sources, embedding
five vector PDF figures. For the official-style, double-blind TMLR PDF and
supplementary ZIP, see the [TMLR build and submission instructions](tmlr/README.md).

## Rebuild

From `transit_hrl/` or the root of the extracted bundle, with Python 3.10+,
GNU Make, Pandoc 3, latexmk/pdfLaTeX, and Poppler (`pdftotext`):

```sh
python3 -m pip install -r paper/requirements.txt
make -C paper reproduce
```

If Pandoc is not on `PATH`, pass `PANDOC=/absolute/path/to/pandoc`.
`make -C paper review-bundle` writes `/tmp/freqhrl-review.tar.gz`. This
analysis package excludes deployment scripts, cluster paths, job receipts,
and archived operational notes. The full local working package is built
with `make -C paper bundle` (`/tmp/freqhrl-submission.tar.gz`).

## Data and scope

| Outputs | Saved inputs | What the rebuild does |
| --- | --- | --- |
| Table 1, Figures 1 and 4 | `paper/data/stage9/replicate_*.json`, eight optimizer roots | Recompute paired root effects and Student-t intervals; use saved call times for Figure 1 |
| Table 8 | PointMaze timing portfolio summary | Check means, contrasts and intervals against the frozen follow-up |
| Table 2, Figure 2a | Label-free query ablation summary (`B = 24`) | Recompute paired eight-root contrasts for clock-grid and observable-change query acquisition |
| Table 3, Figure 5a | PointMaze timing baselines and MLP summary | Verify eight-root contrasts against native periodic, learned binary, distance threshold, cost difference, and raw MLP |
| Tables 4 and 9, Figures 2b-d and 5c,d | Multi-budget and CEM summaries (`B = 12..40`) | Verify same- and cross-budget ISE against periodic and distance timing, plus planner, scoring and measured control time |
| Table 5, Figure 5b | `Reacher-v5` polygonal 8-seed confirmation summary | Verify unfiltered eight-root second-plant confirmation contrasts and waypoint-authority strata |
| Table 10 | Smooth-ellipse Reacher and LQR cart-pole applicability summaries | Check paired applicability contrasts and preserve both diagnostic qualification outcomes |
| Tables 6-7, Figure 3 | Qualified Stage298/301/304 Transit reports and event audit | Preserve single-intervention contrasts and the negative two-decision episode boundary |

The rebuild uses saved outcomes and does not require a simulator or scheduler.
The bundle does not contain the raw trajectories or checkpoints needed to rerun policy training.
