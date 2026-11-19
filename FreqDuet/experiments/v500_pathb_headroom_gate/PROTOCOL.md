# V500: Path B Gate 1 — Multi-Timescale Headroom & Turnaround-Budget Leakage Verification

## Purpose
Before launching 200-episode B1 bi-level constrained SAC training (`v501`), verify on untouched development days (`2600101`–`2600108`) across three corridors (`tidal_bimodal`, `mbta_route111_wkdy`, `flat_sanity`) that:
1. Removing the legacy flat-OD and 900s zero-sum headway handcuffs unlocks substantial physical passenger-exposure and regularity headroom under fixed physical fleet (`fixed_pool`) and strict daily trip conservation (`262` trips/day);
2. Naive unbudgeted station holding (`A3`) suffers cross-timescale turnaround leakage during directional tidal peaks;
3. Coupling causal harmonic low-frequency headway allocation with turnaround-slack-budgeted high-frequency holding (`A4`) jointly improves restricted passenger journey exposure and plan-normalized regularity (`plan_cv2` / `EWT`) over the strong fixed-timetable bilateral baseline (`A1`), while remaining no-harm on `flat_sanity`.

## Arms (5 Deterministic Causal Policies, Exact CRN per Day)
- `A0_fixed_nohold`: Fixed 360 s timetable, zero holding.
- `A1_fixed_bilateral`: Fixed 360 s timetable + causal AVL two-sided holding capped at 15 s/stop.
- `A2_lf_quota_nohold`: Causal APC harmonic low-frequency directional headway allocation with remaining-day trip-quota conservation (`262` trips preserved), zero holding.
- `A3_lf_quota_unbudgeted_hold`: Same `A2` causal LF timetable + unbudgeted two-sided holding up to 45 s/stop (demonstrates turnaround starvation / leakage).
- `A4_lf_quota_turnaround_hf_hold`: Same `A2` causal LF timetable + persistent trip holding budget $B_{v,\text{trip}}$ derived from terminal supply slack + load/HF-aware holding within $u_e \le B_{v,t}$.

## Execution Contract
- Scheduler `priority="high"` + manual `dispatch --task-id` on `node001–node006` only.
- 24 day-corridor shards (`3 corridors × 8 days`) + 1 preflight + 1 aggregate task.
