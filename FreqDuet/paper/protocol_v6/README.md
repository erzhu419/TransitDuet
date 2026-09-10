# Protocol V6 Manuscript Assembly

This directory is generated from the frozen Protocol V6 evidence package by
`freqduet/scripts/assemble_freqduet_protocol_v6_manuscript.py`.

From the repository root, rebuild it with:

```bash
python FreqDuet/freqduet/scripts/assemble_freqduet_protocol_v6_manuscript.py
```

## Contents

- `introduction.md` and `related_work.md`: source-grounded positioning.
- `methods.md`: current method and evaluation protocol.
- `results.md`: evidence-bound main results and two main tables.
- `discussion.md` and `conclusion.md`: interpretation and claim boundaries.
- `availability.md`: evidence, public-data, and archive-status statement.
- `manuscript.md`: complete assembled article draft.
- `supplementary.md`: full outcomes, configuration lineage, statistical
  procedure, negative-result boundary, and external-data provenance.
- `references.bib`: verified working bibliography.
- `terminology.md`: canonical paper terms and prohibited conflations.
- `literature_verification.md`: primary-record bibliography audit.
- `journal_target.md`: target-journal fit and formatting decision.
- `trc_submission/`: flat anonymous `elsarticle` working bundle with separate
  manuscript and Supplementary Material builds.
- `tables/`: standalone Markdown and LaTeX table fragments.
- `figures/`: portable review PNGs for Figures 1-5; publication-format exports
  remain in the frozen evidence package.
- `figure_captions.md`: assembled captions for Figures 1-5.
- `assembly_manifest.json`: source and output inventory.

## Scientific status

The manuscript uses `F_freqduet_protocol_v6_confirmed_main_hiro` as the current best controller. V8
passed its registered 40-episode effect/no-harm gate, although its
Holm-adjusted training-seed sign-flip result is $p=0.125$. V9 did not confirm
the registered 200-episode regularity gate. The V9
fixed-headway comparison is a trade-off, not a passenger-journey superiority
result. The evidence status therefore remains `submission_ready: false`; adding
editorial sections and a journal template does not override that scientific
decision. Author metadata and the current portal-specific submission fields
remain pre-submission tasks.
