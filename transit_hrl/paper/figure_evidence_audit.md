# Core Figures And Remaining Evidence

Date: 2026-10-22. Three figures use completed local results only; no new scheduler
tasks, remote pulls, training or simulation. Renderer: Python/matplotlib.

## Deliverables

| Figure | Scientific role | Exports and source |
| --- | --- | --- |
| 1 | Paired supervision and actual first-registered-path timing | [PNG](figures/fig1_budgeted_replanning.png), [PDF](figures/fig1_budgeted_replanning.pdf), [SVG](figures/fig1_budgeted_replanning.svg), [CSV](figures/fig1_budgeted_replanning_source.csv) |
| 2 | All five independent PointMaze confirmation gates and call/score accounting | [PNG](figures/fig2_pointmaze_confirmation.png), [PDF](figures/fig2_pointmaze_confirmation.pdf), [SVG](figures/fig2_pointmaze_confirmation.svg), [CSV](figures/fig2_pointmaze_confirmation_source.csv) |
| 3 | Transit single-intervention gains versus failed two-decision criterion | [PNG](figures/fig3_transit_boundary.png), [PDF](figures/fig3_transit_boundary.pdf), [SVG](figures/fig3_transit_boundary.svg), [CSV](figures/fig3_transit_boundary_source.csv) |

Rebuild from `transit_hrl/`: `python3 paper/build_plan_validity_evidence.py`,
then `python3 paper/render_plan_validity_figures.py`. The numerical artifact
links each study to its qualified report or original cell summaries. CSV rows
retain raw values, interval endpoints, display scales, n and statistical unit;
source paths are relative to `paper/`, with JSON pointers where applicable.

## Consistency

Figure2 reproduces Table1's root means and original df7 Student-t intervals;
its dots are eight root means, not 128 independent training runs. Scores are
separate operations from planner calls. Figure1 uses saved calls from the first
registered root/path, not a chosen best trajectory or invented timing example.

Figure3 panels a/b reproduce raw-relative contrasts in the saved Stage298/301
reports. Panels c/d/f preserve all three Stage304 parts, including crossing
normal intervals and the opposite cost directions in the two halves. The
half-base criterion is a paired difference, not a ratio-only significance claim.
The caption's unconfirmed joint increment is included in the source CSV.
Fleet decomposition and 315/320 outcome matches are descriptive, not new tests.

Costs use units of 1e-3; waiting/margins are converted from minutes to seconds.
Stage298/301 prefix and conditional-MC intervals stay distinct; Stage304's
bootstrap and normal intervals both quantify complete-episode sampling.
The manuscript captions and evidence map identify these different estimands.

All exports are 183mm wide, with 124/139/190mm heights. Canvas-bound checks and
nonblank-pixel checks accompany rendering; the three PNGs were visually inspected
for clipping and overlap. SVG text remains editable; PDF text extraction checks
that labels are preserved. PNGs are 300dpi previews; PDF/SVG are vector sources.
Nine focused checks pass for original-result/artifact equality, table/CSV
values, timeline provenance, interval types and units, active links and vector text.

## Evidence Still Missing

The figures do not supply competent noncollapsed learned-timing baselines or
representation/capacity-matched history ablations. Shared jitter-trained actors
also do not rank full methods trained under their own timing schedules.

Planner-call equality is not matched total computation or deployment latency;
the score checks, simulator branch acquisition and event-stratified query
selection remain relevant costs and assumptions.

Transit changes the action-selection mechanism and does not provide a second
task using the same fixed timing algorithm. Stage304 remains a negative boundary,
not same-algorithm transfer support or proof that repetition alone caused failure.

Local renewal-value supervision still has no demonstrated episode-level regret
or fleet-risk guarantee. Full [scientific gaps](submission_readiness.md) remain
open; figure completion changes presentation readiness, not those decisions.
