# Plan-validity replanning: review PDF and analysis artifact

[Review PDF](submission.pdf) | [Markdown source](manuscript.md) |
[Evidence map](evidence_map.md) | [Bundle manifest](artifact_files.txt)

The PDF is a venue-neutral, author-neutral draft. Its title, text, tables,
captions and citations come from the Markdown and BibTeX sources. It embeds
the vector figure exports. Apply the selected venue's template, page limits
and anonymity rules before uploading; removing the author block is not a
double-blind audit.

## Rebuild

From `transit_hrl/` or the root of the extracted bundle, with Python 3.10+,
GNU Make, Pandoc 3 (tested 3.9), latexmk/pdfLaTeX (TeX Live with the
recommended and extra LaTeX packages, and Latin Modern) and Poppler
(`pdftotext`) on PATH:

```sh
python3 -m venv /tmp/freqhrl-paper-venv
/tmp/freqhrl-paper-venv/bin/python -m pip install -r paper/requirements.txt
make -C paper reproduce PYTHON=/tmp/freqhrl-paper-venv/bin/python
```

After dependency installation the rebuild is offline. It writes
`paper/plan_validity_evidence.json`, figure CSV/PDF/SVG/PNG files,
`paper/submission.pdf` and intermediate TeX/log files under
`paper/.submission/`, then runs the evidence checks. If Pandoc is not on
PATH, pass `PANDOC=/absolute/path/to/pandoc`. Run `make -C paper bundle`
with the same options to additionally write `/tmp/freqhrl-submission.tar.gz`
(override with an absolute `BUNDLE=/path/to/file.tar.gz`).

## Data and scope

| Outputs | Saved inputs | What the rebuild does |
| --- | --- | --- |
| Table 1, Figures 1-2 | `paper/data/stage9/replicate_*.json`, eight optimizer roots | Recompute paired root effects and Student-t intervals; use saved call times for Figure 1 |
| Table 2 | PointMaze timing portfolio summary | Check means, contrasts and intervals against the frozen follow-up |
| Tables 3-4, Figure 3 | Qualified Stage298/301/304 reports and Stage304 event audit | Preserve published contrasts, paired interval types, and the failed joint criterion |
| Table 5 | Reacher and cart-pole applicability summaries | Check the paired comparisons and preserve both failed qualification decisions |

The Stage9 files retain the outcomes, calls, seeds and fit accounting used by
the builder. Each records its original cell path. In a worktree with the eight
original cell files, `python3 paper/build_plan_validity_evidence.py --extract-stage9`
regenerates these compact inputs. It does not regenerate the underlying
episodes. Follow-up summaries are shipped for audit, not as independent
training replications.

The bundle contains the inputs needed for the offline rebuild, selected
experiment source files for inspection, and frozen protocols. Historical
reports can link to additional files in the full repository. Neither the
bundle nor the rebuild includes raw trajectories, model checkpoints,
training, native simulation or scheduler access. This is a reproduction of
the paper's analyses and presentation, not a rerun of policy learning.
