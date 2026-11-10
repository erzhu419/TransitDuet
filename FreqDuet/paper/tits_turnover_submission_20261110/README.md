# FreqDuet: Evidence-Aligned Submission Materials

Updated 2026-11-11. This separate manuscript preserves all earlier drafts,
experiments and the main policy. Study A--D retain V400--V403's original gates;
Study E adds V404's frozen factorial and operating-condition supplement.

The defensible conclusion is an executable cumulative vehicle-time interface
and measured passenger/circulation/regularity tradeoffs. The new four-condition
panel improves journey and readiness relative to native fixed360 bilateral,
but worsens actual CV2. Budget effects dominate; the present timetable law
does not add consistent service benefits. Learned-supervisor, full-frequency
specialization and preserved-quality superiority over simple limits remain
unestablished. No failed historical gate is relabeled.

## Materials

- Main text: five figures and six tables; supporting information retains the
  development grids, complete factorial/component comparisons, tails, earliness,
  passenger cohorts and detailed interface properties.
- V404: 144 formal paths on 16 new days across four conditions; 146 charged
  starts including engineering. Zero new training. All traffic, resampling and
  plot generation used scheduler high on Linux CPU servers.
- `evidence_manifest.json`: original 37 estimates plus 224 distinct supplemental
  mean/interval records, figure source data and runtime inventory.
- `analysis_sources/`: server readout/plot scripts. Ordinary manuscript rebuilds
  only format already computed results; no local simulation or bootstrap.
- Separate `FreqDuet/runtime_release/v404`: standalone source. Full frozen inputs
  and model files remain server-side. All 144 formal paths used that source;
  relocation passed isolated imports, not a clean dependency reinstall.

```bash
bash build.sh
python3 package.py
```

Outputs: `output/pdf/main.pdf`, `output/pdf/supplementary.pdf` and
`freqduet_turnover_source.zip`. The source ZIP is rebuilt independently after
extraction. Page rendering and log checks are recorded in `build_check.json`
and `visual_qa.json`.

## Remaining Decisions

Author identities, inherited input redistribution rights, licence and a permanent
public deposit must be supplied by the authors. The exact special-issue call
and current submission window have not been verified; the draft uses the general
IEEEtran journal format. One route/fleet, four days per supplemental condition,
unchanged running-noise intensity and unresolved preserved-regularity advantage
remain scientific limitations, not missing values to fill with prose.
