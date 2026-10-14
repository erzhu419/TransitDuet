# Stage236: fixed-candidate stability admission contrasts

## Protocol

- Run: `native_transit_stability_admission_stage236_development_20261015_r1`. Reuse all 32 Stage234 action ledgers and Stage235 diagnostic cells on the same 240-state/eight-future development bank.
- Retain all 27 Stage234 controllers, including raw, cost-MSE and unfiltered mean/safe selection. Add future-only, prefix-only and joint admission for the two anchors and mean/safe selectors (39 controllers total).
- Admit the original fixed candidate only if its original pooled per-source cost/wait improvement remains strict. Future-only checks both inner future halves; prefix-only checks deletion of each validation prefix; joint checks both. Subset cost/wait nonharm allows `1e-12` numerical boundaries; safe selection also requires nonincreased wait-worsening fraction. Subset reselected candidates are ignored.
- Each cell's gate applies to every outer scope/source. Accepted actions/scores equal the old candidate; rejected ones equal raw. Preserve all four scopes, per-source and future-direction outcomes, admission counts and actual action switches. Reproduce all historical controls and Stage235 outer deltas exactly.
- One unbound scheduler CPU task on node001-node006. Zero new fits, optimizer updates, model predictions or native episodes. Pull aggregate JSON only.

## Results

Pending. Compare retained interventions and physical gains against raw and unfiltered controls, including accepted candidates that make no outer action change.

## Limitations

Development-cohort reuse and admission based on correlated OOF subsets do not yield independent performance CI. Prefix deletion removes validation records, not that prefix from all training fits. Returning to raw is abstention, not an algorithm improvement. Reference-refit and prefix-coverage effects remain confounded.
