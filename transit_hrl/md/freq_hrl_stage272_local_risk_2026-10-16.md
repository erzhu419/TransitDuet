# Stage272: Context Interaction versus Local Clipping

t165947/node004: done/0, 108.40s, no retries; 16 tests passed. 12,672 head
fits / 4,224 shared-design solves. No old model refits or native simulations.
Pulled 2.30 MB scalar report only. Stage271 replay is exact; cap excess <1.05e-17.

Main penalty .01, raw prior, MSE change versus raw (negative is better):

| Correction | Prefix OOF / opposite futures | Outer-held / opposite futures |
| --- | ---: | ---: |
| Cached training-RMS fit | +21.04% | +4.67% |
| Same fit, local clip only | +1.35% | -3.31% |
| Standardized linear, clipped | +2.30% | -3.02% |
| Equal-parameter interaction, clipped | +4.07% | +0.81% |

Decision: clipping helps; this interaction map does not beat linear. Both future
directions agree on the clipped old fit: held improves, OOF still worsens. No
risk head/default changes. This is not a general rejection of nonlinear models.

Next Stage273: infer saved prefix54 heads on the same outer-held60 queries as
full60, with frozen priors, normalizers, clipping and both banks. Attribute removal
effects on matching omitted root/regime versus other queries; no new fits,
simulation or gates. Resolve population/composition versus missing-group effects.

Limitations: reused development labels, not seed CI or native policy improvement;
local bounds do not prove no-harm. Interactions are not a linear-model superset.
