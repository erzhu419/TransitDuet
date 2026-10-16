# Stage260: Fixed Calibration Prefix Fits

Frozen path: cached full60 -> prefix54 with full60 calibration -> cached
recalibrated prefix54. Same 16 training futures, prior/basis/cap and old keys.
Only the middle endpoint is fitted; no raw/full-model training or selection.

Freeze reliability, state-balanced GLS precision and rankwise lambda from full60
training labels; slice without subset renormalization. Check cached full KKT
without solving it. Evaluate opposite16 futures after saving new actions/models.
Report fit-population and recalibration effects separately across excluded,
same/other-root retained and held states, including first8/last8.

32 single-core scheduler fits (640 prefix fits / 5,120 solves) and one report;
expand only after complete preflight success. No node binding or local fitting.
Models/calibration/action ledgers stay remote; only scalar summaries pulled.

Limitations: full calibration uses the excluded prefix's training labels, so
this is conditional diagnosis, not strict OOF or a deployable policy. Ordered
path effects are not unique causal shares; RMS/action rates do not add. Reused
development/repeated held exposures are not seed CI. Stage253/248/169 closed.

Next: use the decomposition to decide whether fitting-population sensitivity
or recalibration should be stabilized first; no winner selected on these data.
