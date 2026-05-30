# TransitDuet — Server Deployment Package

## Quick start (one line)

```bash
bash run_all.sh
```

That's it. The launcher auto-detects:
- CPU cores (uses ~1/3 on shared server, half otherwise)
- Free GPU memory (≥2GB → uses GPU, else CPU)
- Already-completed runs (resumes, doesn't re-run)

## Options

```bash
bash run_all.sh --dry-run         # preview plan without running
bash run_all.sh --quick           # 50-ep sanity check
bash run_all.sh --tier 1          # main results only (no ablations)
bash run_all.sh --tier 2          # ablations only
bash run_all.sh --workers 4       # force 4 parallel workers
bash run_all.sh --no-gpu          # force CPU-only
```

## Environment variables

- `TRANSITDUET_SHARED=0`  → assume dedicated server (use half cores instead of 1/3)

## What gets run

**Tier 1 (main results)**: 3 seeds × {A_full, CMA-ES, GA, Fixed} = 12 jobs
- A_full: full TransitDuet v2k with elastic fleet, outputs Pareto frontier
- CMA-ES, GA, Fixed: baseline upper-level methods

**Tier 2 (ablations)**: 3 seeds × {B..G} = 18 jobs
- B: no holding feedback in upper state
- C: no CS-BAPR belief tracker
- D: no hindsight credit (uniform reward)
- E: no MORL (fixed weights)
- F: fixed fleet (no elastic sampling)
- G: no demand stochasticity

Total: 30 jobs, ~2.2h with 10 workers (estimate).

## Output

Each run produces:
- `logs/{name}_seed{N}/diagnostics.csv` — 62-col per-episode metrics
- `logs/{name}_seed{N}/trip_details.csv` — per-trip breakdown (every 25 ep)
- `logs/{name}_seed{N}/history.json` — compact summary
- `logs/A_full_seed{N}/pareto_frontier.json` — fleet∈[8,16] Pareto points
- `logs/{name}_seed{N}/checkpoints/` — model weights

After all runs complete, `scripts/aggregate.py` auto-runs and writes:
- `results/ablation.csv` — ablation study table
- `results/baselines.csv` — baseline comparison
- `results/pareto.csv` — Pareto frontier (mean ± std)
- `results/summary.txt` — human-readable summary

## Dependencies

```bash
pip install -r requirements.txt
```

Tested: Python 3.10, PyTorch 2.0+, NumPy <2.0

## Cost estimates

| Setting | Jobs | Wall clock |
|---------|------|-----------|
| `--quick --tier 1` | 12 | ~5 min |
| `--quick` | 30 | ~10 min |
| `--tier 1` | 12 | ~1h |
| `--tier 2` | 18 | ~1.5h |
| (all) | 30 | ~2.2h |

## Structure

```
transit_duet/
├── run_all.sh              ← entry point
├── runner_v2.py            ← main bi-level RL trainer
├── run_upper_comparison.py ← baselines (CMA-ES, GA, Fixed, ...)
├── config_v2.yaml          ← default hyperparameters
├── configs_ablation/       ← A..G variant configs
├── scripts/
│   ├── launcher.py         ← resource-aware scheduler
│   └── aggregate.py        ← post-processing
├── env/                    ← bus simulation
├── upper/, lower/          ← RL policies
├── coupling/               ← TAP, belief, holding feedback
└── logs/, results/         ← outputs
```
