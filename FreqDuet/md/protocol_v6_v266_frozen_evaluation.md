# V266: frozen S contrasts on the complete E bank

Read the 32 E unit/check/slot summaries and unchanged S seal only. Require all
legal arms, measured execution, shared prefix/future checks and bounded debits;
do not rescan raw trajectories. No reselection, tuning, training or policy change.

V253 statistics: 5000 draws, seed253. Average futures within context before
crossed actor/day resampling. Separately resample paired futures within each
fixed context, following V105's conditional interval construction. Share draws
across all metrics/arms/contrasts. Report full E, fixed halves E0-E3/E4-E7,
actor/day means, S-to-E changes, paired SD/MCSE, signed debiased signal squared
and independent-half cross-products. Do not clip negative signal estimates.

The frozen primary requires full-E passenger CI upper <0 and every original
constraint upper <=0 under BOTH interval types; each half also requires point
passenger improvement and original constraint nonharm, matching V105. Report
both interval types for the halves as well. Native-locked contexts remain exact
zero controls. Other bins are descriptive, never E-selected winners.

Keep complete arm statistics on the server. Return the small primary/axes/noise
and executed-holding/same-bus supply summary. Two actors and two consumed days
only support a finite-context pilot, not a learned frequency-method claim.
