# Core Figures And Traceability Audit

Date: 2026-11-20. Five manuscript figures use saved results only. Renderer: Python/matplotlib.

## Deliverables

| Figure | Scientific role | Exports and source |
| --- | --- | --- |
| 1 | Paired supervision and actual first-registered-path timing | [PNG](figures/fig1_budgeted_replanning.png), [PDF](figures/fig1_budgeted_replanning.pdf), [SVG](figures/fig1_budgeted_replanning.svg), [CSV](figures/fig1_budgeted_replanning_source.csv) |
| 2 | Label-free query ablation (`B = 24`), multi-budget Pareto frontier (`B = 12..40`), distance-cut gains, and cross-budget call savings | [PNG](figures/fig3_unprivileged_and_pareto.png), [PDF](figures/fig3_unprivileged_and_pareto.pdf), [SVG](figures/fig3_unprivileged_and_pareto.svg), [CSV](figures/fig3_unprivileged_and_pareto_source.csv) |
| 3 | Transit single-intervention gains versus failed two-decision episode criterion | [PNG](figures/fig5_transit_boundary.png), [PDF](figures/fig5_transit_boundary.pdf), [SVG](figures/fig5_transit_boundary.svg), [CSV](figures/fig5_transit_boundary_source.csv) |
| 4 | All five independent PointMaze confirmation gates and call/score accounting | [PNG](figures/fig2_pointmaze_confirmation.png), [PDF](figures/fig2_pointmaze_confirmation.pdf), [SVG](figures/fig2_pointmaze_confirmation.svg), [CSV](figures/fig2_pointmaze_confirmation_source.csv) |
| 5 | PointMaze and polygonal Reacher timing contrasts and CEM timing components | [PNG](figures/fig4_route_b_baselines_reacher_cem.png), [PDF](figures/fig4_route_b_baselines_reacher_cem.pdf), [SVG](figures/fig4_route_b_baselines_reacher_cem.svg), [CSV](figures/fig4_route_b_baselines_reacher_cem_source.csv) |

Rebuild from `transit_hrl/`: `make -C paper figures`. The CSV values trace back to [plan_validity_evidence.json](plan_validity_evidence.json). Figure 2d pairs the same roots and paths at different call counts and reports distance as well as periodic comparators. Figure 5d reports planner-only and planner-plus-score times; the latter excludes lower inference. The measured control sum and intervals are reported in the evidence JSON and manuscript.
