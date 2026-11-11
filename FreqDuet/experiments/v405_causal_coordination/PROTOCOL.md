# V405: Causal Holding and Dispatch Coordination

Development screen only. Keep the V404 12-bus fleet, 262 trips, fixed public
360-second anchors and native seven holding commands. Freeze before traffic.
Four new demand days, five physical arms, 20 starts total, no automatic retry
or training. Compare native bilateral, fixed-plan model, frozen 10-s cap,
holding transfer with fixed plans, and holding transfer plus dispatch.

- Replace up-front trip credit cuts with a native no-change reference. Test
  legal current actions and equal predicted command-second exchanges with the
  next controlled stop of the same trip. Admit a transfer only when the first
  command is nonharmful even with native fallback. Recheck the later command
  before execution; log completion or abstention. Equal projected command
  time is not a claim of equal realized all-day holding.
- At a slow publication opportunity, compare the current release with bounded
  public-or-later candidates (0/15/30/60 s) using the same event model. Keep the
  next anchor and immutable dispatch commitments. Evaluate actual modeled
  launches, not changes to scheduled releases alone. The edited release must
  bind relative to the same current holding command without the edit.
  Do not publish when the forecast does not support an improvement.
- Protect forecast passenger exposure, arrival CV2, public delay and readiness
  on the same launched roster. A later terminal return is allowed only through
  physical simulation; it is not treated as free time or extra supply.
- Preserve the closed-block APC forecast where supported. For sparse clearance
  evidence, use the frozen historical OD profile and past APC conservation
  where identifiable. Compare a point scenario and a 25%-higher censored
  demand scenario (with backlog sensitivity when no clearance is observed).
  These are sensitivity assumptions, not statistical or physical upper bounds.
  Missing running evidence or incomplete continuation preserves native action.

Primary development screen: against cap_10, no worse actual CV2, public delay
or readiness on any day, and lower mean journey in both conditions. Compare
the coordinated and fixed-plan transfer arms separately before crediting the
dispatch layer. Read out actual actions, paired-stop fulfillment, earliness,
unserved passengers and censored-case support. Preserve negative results, do
not retune on these days, and do not promote the main policy.
