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
- Completed: `t166106`, node006, done/0, no retries, 1,408 residual fits. Cached predictions, cell metrics and group metrics all reproduce exactly; normal-equation error 2.08e-17. Pulled only 617,234 bytes.

## Results
Mixed-pooled clipped fleet-risk MSE x 1e6 (lower is better).

| Population / labels | Linear9 | Action-context73, matched df | Observed-context73, matched df |
| --- | ---: | ---: | ---: |
| Full60 train / assigned16 | 575.102 | 567.207 | 557.528 |
| Full60 train / opposite16 | 627.827 | 618.782 | 616.460 |
| Prefix54 retained / opposite16 | 628.047 | 617.790 | 615.716 |
| Prefix54 excluded / opposite16 | 670.899 | 670.324 | 659.455 |
| Full60 outer held / opposite16 | 868.048 | 880.862 | 861.331 |
| Same prefix54 heads, outer held / opposite16 | 871.484 | 883.819 | 864.183 |

Mean df changes from 31.839/37.055 to the shared Linear9 target 8.281 (range 7.519-8.824); mean lambda 2.715/4.064. Representation, normalizers, samples and clip were unchanged.
Observed beats matched-df action context by 1.62% excluded / 2.22% held and Linear9 by 1.71% / 0.77%. Against its own Stage277 predictions, held MSE improves 34.253e-6 while train-opposite worsens 8.348e-6: reducing complexity trades interpolation for transfer, rather than improving every error.
Observed clip fractions fall from 72.1% to 30.9% excluded and 67.7% to 23.3% held. Unclipped observed held MSE is 844.096, but excluded MSE 709.690 remains worse than clipped; no clipping change is selected.
Boundaries: mixed-a excluded still loses to matched-df action context by 13.035e-6; root401 excluded/held still lose to Linear9 by 9.396/27.283e-6; mixed outer-held bank1 loses to Linear9 by 2.514e-6. Mixed excluded also remains worse than raw risk by 3.649e-6. Positive pooled prediction comparison, not uniform superiority.
Next Stage279: freeze these heads, substitute only risk into the unchanged raw total-cost selector/feasibility, and evaluate cached opposite16 policy outcomes plus 8+8 halves. No fitting or simulation.
- Limitations: df matches in-sample linear sensitivity, not OOD capacity or correlated simulator-noise amplification. This is development prediction validation, not seed-CI or native control improvement. Stage253/248/169 remain closed.
