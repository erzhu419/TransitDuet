# Core Figures And Traceability Audit

Date: 2026-11-18. Five manuscript figures use completed local results only; no live simulation or remote pulls run during `make -C paper reproduce`. Renderer: Python/matplotlib.

## Deliverables

| Figure | Scientific role | Exports and source |
| --- | --- | --- |
| 1 | Paired supervision and actual first-registered-path timing | [PNG](figures/fig1_budgeted_replanning.png), [PDF](figures/fig1_budgeted_replanning.pdf), [SVG](figures/fig1_budgeted_replanning.svg), [CSV](figures/fig1_budgeted_replanning_source.csv) |
| 2 | All five independent PointMaze confirmation gates and call/score accounting | [PNG](figures/fig2_pointmaze_confirmation.png), [PDF](figures/fig2_pointmaze_confirmation.pdf), [SVG](figures/fig2_pointmaze_confirmation.svg), [CSV](figures/fig2_pointmaze_confirmation_source.csv) |
| 3 | Label-free query ablation (`B = 24`), multi-budget Pareto frontier (`B = 12..40`), distance-cut gains, and cross-budget call savings | [PNG](figures/fig3_unprivileged_and_pareto.png), [PDF](figures/fig3_unprivileged_and_pareto.pdf), [SVG](figures/fig3_unprivileged_and_pareto.svg), [CSV](figures/fig3_unprivileged_and_pareto_source.csv) |
| 4 | PointMaze external baselines & generic MLP, second-plant `Reacher-v5` polygonal 8-seed confirmation, and CEM planner wall-clock decomposition | [PNG](figures/fig4_route_b_baselines_reacher_cem.png), [PDF](figures/fig4_route_b_baselines_reacher_cem.pdf), [SVG](figures/fig4_route_b_baselines_reacher_cem.svg), [CSV](figures/fig4_route_b_baselines_reacher_cem_source.csv) |
| 5 | Transit single-intervention gains versus failed two-decision episode criterion | [PNG](figures/fig5_transit_boundary.png), [PDF](figures/fig5_transit_boundary.pdf), [SVG](figures/fig5_transit_boundary.svg), [CSV](figures/fig5_transit_boundary_source.csv) |

Rebuild from `transit_hrl/`: `make -C paper figures`. Every plotted point and interval in Figures 2--5 is cross-checked to 12 decimal places against [plan_validity_evidence.json](plan_validity_evidence.json) by `tests/test_plan_validity_figures.py`. All five exports are `183 mm` wide, pass strict canvas-bound checks (`text_within_canvas: true`), and preserve editable vector text in PDF and SVG.
