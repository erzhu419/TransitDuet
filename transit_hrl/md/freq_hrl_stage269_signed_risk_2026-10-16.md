# Stage269: Signed Fleet Target Stability

t165770/node006: done/0, 22.79s, no retries; 19 tests passed. 32 cached cells,
240 states, 40 prefixes; no fitting/simulation. Pulled 177,743 scalar bytes only.

Signed probability x amplitude reconstructs means within 2.78e-17. Eligibility
union: 785 state-actions; 392 have no events in either bank. 337/517 observed
action-sign combinations have one-bank support. Raw-label sample variance:
99.844% occurrence, 0.156% amplitude. Bank-gap/noise RMS = 1.003.

Stage268's any-pair KKT within-event scatter includes action/sign-pattern changes;
it is a different estimand. Factorization alone gives no variance gain.

Next: controlled shared-risk-head fitting on cached labels: direct mean regression
vs action-conditioned signed occurrence/amplitude. Match capacity or add an
explicit parameter-count control. Freeze features, sample budget, service path,
reference, actions and selection contract. Train sharing on training labels only;
assess prefix OOF and opposite-bank error in both directions before deployment.

Limitations: reused-label scatter, not true tail risk, seed CI or policy gain.
Empty amplitudes stay unknown; no tail deletion. Stage253/248/169 remain closed.
