# Stage280: selected-pair risk/service margin attribution

- Run: `native_transit_decision_margin_stage280_20261017_r1`.
- Motivation: Stage279 observed-df has lower pooled risk MSE but does not outperform Linear9 on policy cost/wait; fleet savings are offset by service costs. Identify what was predicted, not just what happened.
- Reuse all Stage279 choices and Stage278 predictions. Replay the existing selector exactly, without changing choices, gate, service, clip, sample populations or future assignments.
- Fixed pairs: observed-df versus Linear9/action-df/raw/bounded, plus Linear9 versus raw. Both risk heads are measured on the same pair of saved actions.
- Identity: actual cost margin = predicted total margin + fleet prediction error + service prediction error. Check both controllers' opposite predicted preferences.
- Report signed component means on all query states; risk pair MSE/MC variance conditional on action disagreement; fleet add/save draw counts. No noise subtraction or inferential CI.
- Preserve full train, prefix retained/excluded, full held and prefix-head held scopes; assigned/opposite16 and fixed 8+8 halves. Reproduce all corresponding Stage279 pair contrasts before admission.
- Budget: one scheduler task, 1 CPU / 3072 MiB, any node001-node006 without pin; zero fits, new choices, simulations or actor updates. All 84,480 repeated query-state exposures retained.
- Full margins/half ledgers stay remote. Pull compact scalar report with all16 diagnostics across views/banks and primary mixed held/excluded halves.
- Verification: four focused tests cover exact head-error decomposition, common-pair MSE, no-disagreement populations, cached cell/group replay before any outcome access, compact-report scope and completed-parent scheduler gate; both cold CLI startups checked.
- Completed: `t166128`, node006, done/0, no retries. All 32 cells / 84,480 repeated exposures admitted; raw replay exact, outcome replay error <=2.23e-16 and margin identity error <=3.13e-17. Pulled 882,592 scalar bytes (166,645 transport bytes), no query ledgers or models.

## Results And Next
Mixed-pooled observed-df minus Linear9 on identical frozen action pairs; cost units x1e6, negative is better.

| Population | Predicted total | Actual total | Predicted fleet | Actual fleet | Predicted service | Actual service |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train / assigned16 | -346.731 | +49.232 | -353.367 | +37.977 | +6.636 | +11.255 |
| Train / opposite16 | -346.731 | -134.712 | -353.367 | -170.898 | +6.636 | +36.187 |
| Prefix excluded / opposite16 | -380.205 | +1.318 | -399.900 | -32.552 | +19.694 | +33.870 |
| Outer held / opposite16 | -298.648 | +48.390 | -294.263 | -27.127 | -4.385 | +75.517 |

Fleet error explains 96.3% / 77.0% of the predicted-to-actual cost gap on excluded / held; service error explains the rest. Excluded service sacrifice was already predicted, while held service changes sign. This is an accounting attribution, not two separately identified causal defects.
On these disagreement pairs, risk MSE x1e6 is observed/Linear9 275.243/290.634 excluded but 271.708/264.582 held. The held all-action MSE advantage does not transfer to the selected pairs. Only 90 / 86 nonzero fleet draw contrasts occur among 3,904 / 3,552 disagreement-draw exposures; repeats are not independent events or seeds. MC variance-of-mean is 234.935 / 240.703, so this does not identify a population MSE or irreducible noise floor.
Both opposite-future halves worsen service/wait, but total-cost signs differ. Held service/wait harm also agrees across both banks and both mixed views; excluded service/wait differs across banks. Raw comparison remains distinct: observed is +173.864e-6 cost / +0.000206608 min wait excluded, but -366.298e-6 / -0.000706380 min held.
Next Stage281: cross future-bank component replacement with finite-MC fleet-only, service-only and joint estimates, fixed feasibility and all frozen controls. Use assigned16 or its fixed 8-draw halves for proposals; evaluate opposite16 and its halves. This diagnostic separates the usefulness of better component information without another fit, gate search or simulation; finite-MC replacements are not an oracle upper bound or a deployable policy.
- Limitations: descriptive diagnosis on reused development futures and fixed single-intervention action pairs; not causal attribution, independent training-seed evidence or full HRL validation. No deployment/default change; Stage253/248/169 remain closed.
