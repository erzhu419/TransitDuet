# Plan-Validity Figure Contract

Date: 2026-10-22. All three figures are rendered in `figures/` as editable
SVG, PDF and 300dpi PNG, each with source CSV. Backend: Python/matplotlib.
The archived MuJoCo/Quant responsibility figures belong to the previous paper
argument. Numerical source: `plan_validity_evidence.json`; renderer:
`render_plan_validity_figures.py`. No new training or simulation is involved.

Export width: 183mm. Heights: 124mm / 139mm / 190mm. White backgrounds,
consistent sans-serif text, 6.1-9pt labels, distinct point shapes for CI types.

## Figure 1: Timing rule and acquisition

A 50-step-bin timeline from the first registered confirmation root/path
(209011/2180301), not selected by performance, shows periodic, random-offset and learned-threshold
calls, the offset-25 deadline and variable 25-75-step option durations. A separate
training branch shows event-stratified simulator queries, paired keep/renew
labels and the fitted 39-feature predictor. Deployment has causal observations
and no branch simulation. The caption distinguishes call count from score checks.

## Figure 2: Independent PointMaze timing evidence

Use the five registered contrasts and eight root effects in
`plan_validity_evidence.json`. Display optimizer roots as points and the original
paired-root Student-t CI as the summary. Distinguish the primary ISE contrast
from the four required secondary gates. Show 24 calls for every mode and
84.1875 candidate score checks separately. Development results are not pooled.

## Figure 3: Transit positive effects and failed repeated-decision criterion

Use separate panels for Stage298/301 single-intervention outcomes and Stage304
complete-episode outcomes. Labels name the different uncertainty sources:
prefix versus conditional MC in the former; episode bootstrap versus paired
normal in the latter. Include total cost, waiting, the fleet/service split,
both 16-scene-per-stratum halves (160 scenes each) and the failed paired
half-base retention margin. Cost is displayed in units of 1e-3; waiting and
waiting margins are converted from minutes to seconds.
No connecting trajectory may imply that these are matched samples of one
algorithm. The four fleet events and 315/320 joint/service outcome matches
may be annotated as descriptive counts from the existing event audit.

## Supplement

Historical learned-upper and responsibility results can have a separate source
map; their algorithms and statistical units stay visible. Figure sources are
the active evidence artifact and linked reports, not the old figure builder's
MuJoCo/Quant registry selection. The three core exports and source data are
complete; venue-specific formatting and the broader scientific gaps remain
separate work. See `figure_evidence_audit.md` for the numerical and visual checks.
