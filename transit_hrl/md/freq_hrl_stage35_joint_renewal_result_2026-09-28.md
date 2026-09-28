# Stage-35 Joint Renewal Result

All 32 tasks `t102121`-`t102152` completed on node001-node006: eight roots,
four treatments, 1024 evaluation episodes. All 32 trajectory audits and
selected-checkpoint replays pass. Independent root-count bootstrap matches.

**Decision: `stage35_development_gate_failed`, 1/5 endpoints supported.**

| Learned-history contrast | Mean | Adjusted CI |
|---|---:|---:|
| Return increase vs fixed50 | -10.4185 | [-33.2309, 12.3045] |
| ISE reduction vs fixed50 | -0.34426 | [-0.68428, -0.04264] |
| Planning-call savings vs fixed50 | -4.9375 | [-8.8502, -1.2109] |
| Utility increase vs fixed100 | 51.7393 | [37.0887, 67.5173] |
| Utility increase vs learned-current | 1.0653 | [-5.0400, 5.9741] |

History return is 902.29 versus fixed50 912.71; ISE 1.80077 versus 1.45651
(+23.64%); calls 28.94 versus 24 (+20.57%). All 16 learned treatments select
iteration0, despite nonzero actor/critic training. Their deployed policies
are initialization, not a successful learned-renewal result. History selection
return falls from 916.38 initially to 848.20 at iteration128 while calls fall
from 30.30 to 22.52. This is training/adaptation failure, not inadequate seeds.

Method cost: 42124800 steps, 752124 upper calls, 712946 gate calls, no previews.
Verification: 38400 steps, 634 upper calls, 636 gate calls. Mean training wall
time is 324s for history and 298s for fixed50; timing is descriptive. Only
compact JSON returns locally. Twenty-one focused tests pass, including a fix that
reads each compressed trajectory array once rather than repeatedly decoding.

## Limitations And Next

Fixed100's positive contrast does not establish learned-policy or frequency
benefit. Final trained weights were not retained in this run; future runs now
save final and selected checkpoints server-side. Next isolate gate-only,
upper/lower adaptation and joint updates on fresh development paths before
changing the credit estimator. Earlier negative results remain unchanged.
