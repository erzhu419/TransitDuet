# State-Dependent Bus Holding

This is the active FreqDuet manuscript. The main result concerns capped V406
holding under a fixed fleet and timetable. The comparison matches aggregate
commanded holding and its action-bin distribution; it is not evidence for a
learned supervisor or for frequency separation.

- [Manuscript](output/pdf/main.pdf)
- [Supplement](output/pdf/supplementary.pdf)
- [Source package](freqduet_capped_holding_source.zip)

The earlier [turnover manuscript](../tits_turnover_submission_20261110/README.md)
remains a historical snapshot, including its figures and broader experiment
record. This rewrite changes no controller or experiment.

## Evidence Map

| Manuscript use | Frozen protocol |
| --- | --- |
| Main table: within-day comparison | [V410](../../experiments/v410_within_day_control/PROTOCOL.md) |
| Main table: demand transfer | [V411](../../experiments/v411_demand_transfer/PROTOCOL.md) |
| Main table: speed transfer | [V412](../../experiments/v412_running_speed_transfer/PROTOCOL.md) |
| Supplement: whole-day mixture | [V409](../../experiments/v409_holding_dose/PROTOCOL.md) |
| Supplement: guarded forecast | [V418](../../experiments/v418_spacing_confirmation/PROTOCOL.md) |
| Supplement: historical budget and supervision | [V400](../../experiments/v400_independent_confirmation/PROTOCOL.md) |

The [frozen controller](../../experiments/v407_service_continuation/frozen_v406.py)
and [10 s wrapper](../../experiments/v409_holding_dose/study409.py) identify
the tested implementation. The experiment directories contain their frozen
plans, random seeds and path-level summaries.

## Rebuild

Run `bash build.sh` from this directory or the extracted source package. It
requires Python 3, ripgrep and TeX Live with latexmk, PGFPlots and the IEEEtran BibTeX style.
The small JSON files in `evidence/` are copies of the archived paired-day
summaries. `summarize.py` verifies the aggregate gates and regenerates the
table rows without running simulations or resampling. In the repository,
`python3 summarize.py --snapshot` refreshes the JSON copies from the original
run directories. The source package rebuilds the manuscript and displays its
evidence; it does not contain the simulator or checkpoints.
