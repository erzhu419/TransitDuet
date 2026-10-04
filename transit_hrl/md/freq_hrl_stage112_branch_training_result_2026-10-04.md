# Stage112 Branch Training Result

## Protocol

Stage112 trained only a zero-initialized optional action residual above the qualified Stage111 blind lower donor. The base actor, upper policy, values, standard deviation, critic and optimizer state were frozen. Eight roots, two periods, three training arms, 48 out of 48 expected final branch checkpoints, and 3,072 fresh evaluation episodes were completed under the preregistered protocol.

The full run used 52,224 native episodes and 62,668,800 lower steps. Raw checkpoints and traces remain on the server; this checkout contains only compact scalar evidence.

## Result

The mechanical gate passed. Learned residual advice improved over blind residual advice at both periods:

- period 50: mean `+0.005752`, corrected CI `[+0.003603, +0.007913]`;
- period 100: mean `+0.014894`, corrected CI `[+0.007110, +0.025333]`.

The branch gain gate required learned to beat blind, causal forecast, and learned-advice-blinded controls at both periods. It was not supported:

- learned minus forecast, period 50: mean `-0.00000367`, CI `[-0.00001807, +0.00000484]`, inconclusive;
- learned minus forecast, period 100: mean `-0.00001410`, CI `[-0.00004049, -0.00000017]`, negative.

The learned residual therefore adds useful action correction over blind advice, but the learned plan does not outperform the causal forecast. Upper learning is withheld. This is a valid negative result, not evidence for end-to-end learned promotion or full actor-critic HRL.

## Next Step

Do not reopen upper training. First run a compact frozen diagnostic comparing learned and forecast plan/action contexts and their residual readout outputs under the same paired paths. The purpose is to determine whether the failure is indistinguishable plan context, residual collapse to the forecast branch, or a genuinely inferior learned plan. Only a pre-registered diagnostic-supported change should justify another training cohort.
