# Protocol V6 Manuscript Assembly

This directory is generated from the frozen Protocol V6 evidence package by
`freqduet/scripts/assemble_freqduet_protocol_v6_manuscript.py`.

From the repository root, rebuild it with:

```bash
python FreqDuet/freqduet/scripts/assemble_freqduet_protocol_v6_manuscript.py
```

## Contents

- `methods.md`: current method and evaluation protocol.
- `results.md`: evidence-bound main results and two main tables.
- `manuscript.md`: assembled title, abstract, Methods, and Results.
- `supplementary.md`: full outcomes, configuration lineage, statistical
  procedure, negative-result boundary, and external-data provenance.
- `tables/`: standalone Markdown and LaTeX table fragments.
- `figures/`: portable review PNGs for Figures 1-5; publication-format exports
  remain in the frozen evidence package.
- `figure_captions.md`: assembled captions for Figures 1-5.
- `assembly_manifest.json`: source and output inventory.

## Scientific status

The manuscript uses `F_freqduet_protocol_v6_confirmed_main_hiro` as the current best controller. V8
confirmed its 40-episode headway-regularity effect with passenger-journey
no-harm. V9 did not confirm the registered 200-episode regularity gate. The V9
fixed-headway comparison is a trade-off, not a passenger-journey superiority
result. The package therefore remains on submission hold.

The present assembly covers Methods, Results, tables, captions, and
Supplementary Material. Introduction/Related Work citations, author metadata,
and a target-journal template remain separate editorial tasks; they must not
change the empirical wording or the V8/V9 decision.
