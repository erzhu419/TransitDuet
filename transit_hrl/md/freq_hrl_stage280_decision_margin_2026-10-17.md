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
- Status: submitted as `t166128`; source `03a5744165` pushed. Completion and results pending.
- Limitations: descriptive diagnosis on reused development futures and fixed single-intervention action pairs; not causal attribution, independent training-seed evidence or full HRL validation. No deployment/default change; Stage253/248/169 remain closed.
