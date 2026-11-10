# FreqDuet frozen traffic release

This standalone source/input snapshot ran the V404 frozen nine-arm, four-condition
study. No controller source is changed by this packaging step. The source-only
archive omits the native input spreadsheets and model/checkpoint assets.

Use Python 3.10 in an independent CPU environment. `requirements-runtime.txt`
contains installed version pins; use the official CPU PyTorch wheel index for
torch. `requirements-frozen.txt` is the original conda-environment inventory,
not a pip recipe because some entries contain build-machine URLs. Source-path
isolation was tested in the dedicated existing environment; a clean reinstall
has not been tested.

```bash
python -I run.py import_check
python -I run.py case --root /path/to/fresh/output --regime nominal --day 2440001 --arm pressure
```

Run traffic only on scheduler-managed servers at high priority. This example
day is consumed; another execution is reproduction, not an independent sample.
The frozen design is in `python/protocol.json`. The source manifest records
the actual frozen module origins. Manuscript tables, figures and their compact
source data are supplied in the separate paper package.

The archive runs offline and needs no transit API credential. A technical
archive does not settle inherited input-data redistribution permissions, the
software licence, author details or a public repository deposit.
