# Boundary Training Information Versus Coverage

Freeze parent-scaled 120-column potential, DF 3, C0 and sampling rules. Reuse
the eight old prefixes' four windows/check and A0/1 only; exposed B2/3 never fit.
No loss/scale/capacity search. Both strategies pay 1,411,200 NEW native steps/root:

| Strategy | Training prefixes | New work/root | Total A pairs |
| --- | ---: | --- | ---: |
| Dense | 8 existing | 588 new suffix pairs | 780 |
| Wide | 8 existing + 24 new | 576 new pairs + 24 factual episodes | 768 |

Dense adds indices 4-9 to all 96 windows and index 10 to 12 outcome-independent,
balanced windows. Wide new prefixes use indices 4/5 and one factual episode to
select three reachable checks; mechanical prechecks establish the replay path,
so no redundant original-arm episodes are included in formal sampling.
Complete 1,200-step replays are charged, never just suffix length. This matches
incremental simulator cost, not label count or measured total CPU.

Reserve eight NEW evaluation prefixes/root with four windows/check and E20-27.
Freeze all eight dense/wide models before any evaluation access; reuse frozen
base, constant, causal-t and zero controls without controller/raw-reference
training. Report paired tail-MSE differences, prefix and conditional MC errors,
E4 halves and window halves separately per root; report both strategies, no
held-data winner selection. Evaluation costs 1,852,800 steps/root.

## Current Release

Only four mechanical prechecks: cached-window extension plus a dedicated new
prefix, two windows/two suffixes each. Indices 1000/1001 and precheck prefixes
are excluded from training/evaluation. Each root pays 19 full episodes=22,800
steps; total 91,200, zero fits. Precheck factual C0 replay, causal-prefix equality,
cached/fixed boundary contexts and W+tail=G. Raw banks/models stay server-side.
All new roles exclude historical, reserved and effective controller-training seeds.
Formal sampling/fitting/readout needs a separate next-stage release.
Dense allocation's `state_index` means ascending opportunity step within each
prefix; `window_index` is the saved 0-3 window order.

## Limits

Old prefixes and B have informed this design; only reserved readout is fresh.
Boundary prediction is label-side, not a deployable earlier-time policy or an
episode-level HRL claim. CPU is measured, not assumed matched; lifecycle costs
remain incomplete. Planned formal cost is 18,700,800 steps, or 18,792,000 with
prechecks; none of that formal budget is released now. HOLD remains.

## Precheck Result

Code `d9cc7c66b7`; 12 focused tests passed. Tasks t214612-t214615 completed
on node004/node005, all four native paths passed. Exact total: 91,200 steps,
zero fits and 84.158 measured CPU seconds. Only 8,829 public bytes returned.
Cached contexts/costs, fresh factual replay, causal-prefix equality and W+tail
reconstruction passed; reserved training/readout prefixes remain untouched.
See [summary](../results/plan_validity_boundary_information_20261111/precheck_summary.json)
and [receipt](../results/plan_validity_boundary_information_20261111/precheck_release.json).
Next release is training acquisition only: two Dense plus six Wide shards/root,
32 jobs/11,289,600 steps total. No fitting or E readout follows automatically.
