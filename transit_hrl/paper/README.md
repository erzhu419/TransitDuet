# Plan-validity replanning: review PDF and analysis artifact

[Review PDF](submission.pdf) | [Markdown source](manuscript.md) |
[Evidence map](evidence_map.md) | [Readiness summary](submission_readiness.md) | [Bundle manifest](artifact_files.txt)

The PDF is a venue-neutral, author-neutral draft. Its title, text, tables,
captions and citations come from the Markdown and BibTeX sources, embedding
five vector PDF figures. Apply the selected venue's template, page limits
and anonymity rules before uploading.

## Rebuild

From `transit_hrl/` or the root of the extracted bundle, with Python 3.10+,
GNU Make, Pandoc 3, latexmk/pdfLaTeX, and Poppler (`pdftotext`):

```sh
python3 -m pip install -r paper/requirements.txt
make -C paper reproduce
```

If Pandoc is not on `PATH`, pass `PANDOC=/absolute/path/to/pandoc`. Run
`make -C paper bundle` with the same options to write `/tmp/freqhrl-submission.tar.gz`
(override with `BUNDLE=/path/to/file.tar.gz`).

## Data and scope

| Outputs | Saved inputs | What the rebuild does |
| --- | --- | --- |
| Table 1, Figures 1-2 | `paper/data/stage9/replicate_*.json`, eight optimizer roots | Recompute paired root effects and Student-t intervals; use saved call times for Figure 1 |
| Table 2 | PointMaze timing portfolio summary | Check means, contrasts and intervals against the frozen follow-up |
| Table 3, Figure 3a | Label-free query ablation summary (`B = 24`) | Recompute paired eight-root contrasts for clock-grid and observable-change query acquisition |
| Table 4, Figure 4a | PointMaze timing baselines and MLP summary | Verify eight-root contrasts against native periodic, learned binary, distance threshold, cost difference, and raw MLP |
| Tables 5-6, Figures 3b-d & 4c,d | Multi-budget and CEM summaries (`B = 12..40`) | Verify ISE, call counts, and planner, scoring and measured control timing contrasts |
| Table 7, Figure 4b | `Reacher-v5` polygonal 8-seed confirmation summary | Verify unfiltered eight-root second-plant confirmation contrasts and waypoint-authority strata |
| Table 8 | Smooth-ellipse Reacher and LQR cart-pole applicability summaries | Check paired applicability contrasts and preserve both diagnostic qualification outcomes |
| Tables 9-10, Figure 5 | Qualified Stage298/301/304 Transit reports and event audit | Preserve single-intervention contrasts and the negative two-decision episode boundary |

The rebuild uses saved outcomes and does not require a simulator or scheduler.
The bundle does not contain the raw trajectories or checkpoints needed to rerun policy training.
