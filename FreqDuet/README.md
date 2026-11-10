# FreqDuet

## Current Research Entry

The evidence-aligned manuscript and study map are indexed in
[`paper/CURRENT.md`](paper/CURRENT.md). The current traffic-control study uses
bounded timetable revisions, persistent trip budgets and seven-bin model-based
holding. Learned publication permission and harmonic inputs have separate
experimental evidence; the historical HRL description below is not a statement
that the current controller jointly learns both layers.

The standalone frozen source is in [`runtime_release/v404/`](runtime_release/v404/README.md),
with full input/model artifacts retained on the research server. The bounded
factorial and demand-condition supplement is in
[`experiments/v404_publication_closure/`](experiments/v404_publication_closure/PROTOCOL.md).
Traffic runs, analysis and plotting belong on scheduler-managed CPU servers.
Old controls, negative outcomes and earlier manuscripts remain historical records.

## Historical Module Overview

`freqduet/` is an isolated copy of the active `transit_duet` code. It excludes
logs, caches, and archives. All experimental changes for frequency-separated
hierarchical RL should happen here, not in the original `transit_duet/` folder.

Implementation slice:

- causal demand-frequency tracker in `freqduet/frequency/`
- current intensity candidate is a causal dynamic harmonic smoother initialized
  from historical OD-table priors; online RLS adapts it to each episode
- causal Haar-style trailing-window multiscale filtering remains the robust
  wavelet-style baseline
- EMA remains as an explicit baseline config
- RawHistory baseline exposes trailing realized-demand bins without splitting
- frequency allocation is configurable (`upper_mode: low|high|all`,
  `lower_mode: high|low|all`) for split, all-frequency, and swapped ablations
- realized passenger arrivals feed the tracker online from `env/station.py`
- upper state replaces raw hourly demand with low-frequency realized demand and
  appends low-slope, low-forecast, and high-energy features
- realized OD arrivals are tracked online; the upper state also receives compact
  OD-structure features instead of collapsing demand to a single total only
- lower state appends station-local high-frequency residual, residual change,
  local high-energy, and global high-energy features
- lower cumulative holding drift and upper high-frequency delta penalties are
  available through `leakage`
- diagnostics now include upper high-frequency action power, lower low-frequency
  cumulative-drift ratio, and a demand-action attribution proxy
- timetable-curve config entry point:
  `freqduet/configs_freqduet/F_freqduet_timetable_hiro.yaml`
  This is the dev-manual MVP path: the upper writes a rolling low-frequency
  target-headway plan every 15 minutes while launch times stay fixed; the
  stable default uses an asymmetric headway-adjustment range to avoid sustained
  wait-increasing slack. Real terminal dispatch retiming is a later environment
  change.
- main intensity-only config entry point:
  `freqduet/configs_freqduet/F_freqduet_harmonic_hiro.yaml`
- Haar baseline entry point: `freqduet/configs_freqduet/F_freqduet_haar_hiro.yaml`
- EMA baseline entry point: `freqduet/configs_freqduet/F_freqduet_ema_hiro.yaml`
- ablation configs live under `freqduet/configs_freqduet/F_*.yaml`

Historical smoke command (server only):

```bash
cd FreqDuet/freqduet
python3 runner_v3.py --config configs_freqduet/F_freqduet_timetable_hiro.yaml --episodes 1 --seed 42
```

Historical module test (server only):

```bash
cd FreqDuet/freqduet
python3 scripts/eval_frequency_modules.py --n-seeds 5
```
