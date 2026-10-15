# Stage248: frozen-policy conditional-future confirmation

Stage247 state-balanced GLS improved mixed held-prefix cost/wait on eight reused
development futures, but mixed_b and dedicated single-root models still harmed.
The next experiment freezes actions rather than retuning against those outcomes.

## Protocol

- Compare frozen full-model held60 raw, all-MSE, pair-GLS and state-GLS actions.
- Preserve all 32 cells, four views and the existing 240 states/40 prefixes.
- Seal divergent state/action unions before any new outcome; same-action contrasts
  stay exactly zero in the original full-cohort denominators.
- Use 64 new independent future seeds/state, split 32+32, CRN across actions.
- Seed = 1480000000 + Stage195 state index * 64 + draw; no adjacent-probe overlap.
- Reproduce the original source, then resampled observer and original identity;
  replay only the sealed candidate union, with original-greedy continuation.
- Require unchanged prefix/RNG/context, frozen weights, physical clock and ledger.
- No raw/reference fitting, critic updates, action inference or reselection.
- Primary: mixed-pooled cost and wait versus raw, conditional normal MC upper 95%
  bounds below zero with negative means in both 32-draw halves.
- Paired comparisons with all-MSE/pair-GLS and root/view/regime results are secondary.

## Execution

Scheduler only, node001-node006, one core/task, no node binding.
Seal -> ten native preflight cells -> qualified preflight report -> remaining
draws -> final report. Raw labels/rollouts remain server-only; sync small receipts.

## Limitations

This confirms conditional-future replication on existing development prefixes,
not new training-seed CI, unobserved fleet-tail guarantees, or complete end-to-end
learned Freq-HRL. Default and deployment eligibility remain unchanged.
