# Server analysis entrypoints

These are the actual Python entrypoints for the empirical figures, component
comparisons and service/circulation readouts. They use the frozen CPU dependency
versions, the standalone runtime's `worker`/support modules, and the retained
server archive layout recorded in the experiment manifest. `study.py` supplies
the shared JSON helpers and formal-case aggregation.

For example, in the dedicated server environment:

```bash
ARCHIVE=/path/to/v404/run01
VIEW=/path/to/fresh/reanalysis/run01
mkdir -p "$VIEW"
ln -s "$ARCHIVE/cases" "$ARCHIVE/engineering" "$ARCHIVE/aggregate" "$VIEW/"
export PYTHONPATH=/path/to/freqduet_runtime/python:/path/to/analysis_sources
python analysis_sources/archive_figures.py --root "$VIEW" --assets /path/to/freqduet_runtime/assets supplement
python analysis_sources/extra_readout.py --root "$VIEW" --assets /path/to/freqduet_runtime/assets
python analysis_sources/finish_figures.py --root "$VIEW"
```

Schedule analysis on the server. The fresh view reuses retained inputs and
creates new `figures`/`extra` outputs; it does not overwrite archived results.
The entrypoints preserve four-day condition panels and original daily-mixture
probabilities. They do not select another policy or retrain a controller.
The older calibration and retention figures additionally read their named
V400/V402/V403 sibling archives. Those raw archives are retained server-side,
not included in the manuscript ZIP.

The manuscript's ordinary `build.sh` does not execute these scripts. It only
formats supplied estimates and compiles supplied vector figures. Figure source
JSON and editable PDF/SVG assets are included for inspection and reuse.
