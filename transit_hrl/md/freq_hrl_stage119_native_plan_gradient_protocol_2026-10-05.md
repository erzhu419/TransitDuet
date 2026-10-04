# Stage118 Result And Stage119 Native Plan Gradient

Stage118 full (`t135464`-`t135472`) completed mechanically but did not support its gain gate.

| Period | Suffix minus forecast | Bonferroni4 CI | Suffix minus option | Bonferroni4 CI |
| --- | --- | --- | --- | --- |
| 50 | -0.000066315 | [-0.000328297, 0.000174768] | -0.000078900 | [-0.000426159, 0.000253082] |
| 100 | -0.000026468 | [-0.000378252, 0.000319591] | -0.000174036 | [-0.000350769, 0.000024202] |

Exploratory training diagnostics: suffix cross-batch gradient cosine averaged 0.003536
and 0.015994 (32/64 and 30/64 positive); paired credit RMS averaged 1.0522 and 1.8644.
The upper already observes physical state and tracking error. Stage117's local plan
shift RMS was about 0.00715, but its actual option command response was only
0.00000767/0.00000326. These observations point to weak control authority and unreliable
score-gradient estimation; they do not establish a representation impossibility.

Stage119 retains the full eight-coordinate actor, decoder, learned lower, forecaster,
fixed clocks, standard deviations and eight final updates. At each fresh causal query,
intervene in one current upper mean coordinate by +/-0.05 for that option only; all
other decisions execute the current deterministic upper mean. Pair prefix and lower
innovations within each of two independent suffix-noise panels. Use complete native
suffix-return differences, then pull those action derivatives through the actor mean.
Average all queries/panels and use the existing Fisher radius. No direction filtering.

Cover every upper decision, including the first and last. Twelve queries/update/period,
eight roots; 54,272 native episodes and 65,126,400 steps. Preflight: 304 300-step episodes,
91,200 steps, mechanical only. Eight native workers/root, nine declared CPUs, 8 GiB RAM;
dynamic node001-node006 placement. No checkpoints or native arrays return locally.

Final evaluation: 32 fresh scenarios/root/period; common lower noise; compare native FD
with forecast and the frozen final Stage118 suffix upper. Blinding is an execution check,
not another endpoint. Four corrected CIs, bootstrap65,536; require both contrasts positive
at both periods. Deployment uses a causal policy forward, never future branch search.

## Limitations

Training has more native simulation than Stage118 and uses deterministic upper means;
this is not an equal-budget single-factor superiority test. The target is learned upper
gain transfer, not full joint actor-critic, learned promotion, or practical materiality.
