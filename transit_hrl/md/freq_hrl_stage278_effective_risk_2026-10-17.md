# Stage278: feature-only effective complexity control

- Run: `native_transit_effective_risk_stage278_20261017_r1`.
- Motivation: Stage277 true-state73 improves mixed-pooled same-capacity comparisons but loses to Linear9 on both primary held populations; interpolation, bank sensitivity and extrapolation remain distinct problems.
- Keep both cached 73-coefficient maps and all context/lift normalizers unchanged. Keep raw reference, priors, feature basis, samples, assigned16, eligibility, original anchors and clip radii unchanged.
- Treatment: for each fit subset, compute Linear9 effective df at lambda=0.01. Choose each lifted map's lambda >= 0.01 to match that df, using training features only; then fit its residual once.
- For state-balanced centered design Gram eigenvalues s, df = 1 + sum(s / (s + lambda)); intercept unpenalized. No label-based penalty search or held-set selection.
- Controls: cached Stage277 fixed-lambda73, Linear9, bounded residual and raw predictions. Reproduce both old maps and all seven old arms' error sums before admission.
- Evaluate the unchanged train/retained/excluded/full60-held/prefix54-held populations against same/opposite16. Primary: mixed-pooled excluded/opposite and held/opposite; report all views/banks without changing the selector.
- Budget: one unpinned scheduler task, 1 CPU / 3072 MiB, any node001-node006; 1,408 residual fits, no raw/prior refit, simulation, action selection or actor update.
- Read caches remotely; pull scalar report only.
- Verification: five focused tests, including weighted hat-trace identity, label-independent penalties, exact df matching, clip/original anchors, cached-cell/group replay and completed-parent gate; both cold CLI startups passed.
- Status: source ready; scheduler submission pending.
- Limitations: df matches in-sample linear sensitivity, not OOD capacity or correlated simulator-noise amplification. This is development prediction validation, not seed-CI or native control improvement. Stage253/248/169 remain closed.
