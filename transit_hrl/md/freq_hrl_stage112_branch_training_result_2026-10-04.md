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

## Diagnostic

A frozen follow-up compared the final learned and forecast branches on 32 fresh paired episodes per period. Advice cosine was `0.998853` at period 50 and `0.998721` at period 100. Advice norms and residual correction norms were also effectively identical. Learned minus forecast return was `+0.00000358` with CI `[-0.00000091, +0.00000805]` at period 50 and `+0.00000604` with CI `[-0.00000614, +0.00001920]` at period 100.

The current learned branch is therefore operationally forecast-equivalent in this protocol. The residual learner is active relative to blind advice, but the learned plan does not provide a distinguishable control signal.

## Next Step

Do not reopen upper training or claim end-to-end promotion. Any next cohort must first change the plan representation or training signal so learned and forecast plans are causally distinguishable, then preregister a new ablation. Tuning the gate or adding seeds to this collapsed branch is not justified.
