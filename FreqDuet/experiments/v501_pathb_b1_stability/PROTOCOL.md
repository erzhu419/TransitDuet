# V501 Gate 2 Protocol — FreqDuet-v7 (B1: Pure Bi-Level Constrained SAC) & Long-Horizon Stability

## 1. Objective
Validate that genuine **Pure Bi-Level Constrained Soft Actor-Critic (`FreqDuet-v7` / Paradigm B1)** with **Frequency-Separated Observation (`Upper-LF + Lower-HF`)** and **Turnaround-Budgeted Coordination**:
1. Substantially outperforms both `A0_fixed_nohold` and `A1_fixed_bilateral` on both non-stationary corridors (`tidal_bimodal` and `mbta_route111_wkdy`) under matched fixed vehicle pools (`N_fleet`) and exact daily trip quotas (`262` trips/day);
2. Remains **monotonically stable across long-horizon bi-level training (`ep40 -> ep100 -> ep200`)**, eliminating the late-training turnaround collapse (`J_200 > J_40`) observed in legacy unbudgeted/unmasked HRL (`V1–V418`).

## 2. Strictly Separated Scenario Splits
- **Training episodes (`ep = 1..200`)**: `scenario_seed = 1000000 + seed * 10000 + ep` across `8` independent RL seeds (`101..108`) per corridor.
- **Gate 2 Validation CRN Days (`12` days)**: `2650101..2650112` (strictly disjoint from Gate 1 dev days `2600101..2600108` and reserved Gate 3 confirmation days `2700101..2700116`).
- **Frozen Checkpoints Evaluated on All 12 Validation Days**: `ep40`, `ep100`, `ep200`.

## 3. Pure Bi-Level Constrained SAC Formulation (`FreqDuet-v7`)
- **Estimator-Free Physical Rewards**: Neither $r^U$ nor $r^L$ references any `DemandFrequencyTracker` variable; frequency separation acts strictly through causal observation channels (`aligned`, `raw`, `all_band`, `swapped`, `nofreq`), ensuring zero reward-shaping bias in Gate 3's 4-way frequency ablation.
- **Upper Slow-Clock Ensemble SAC ($\Delta T_U = 900\text{ s}$)**:
  - State $s^U_m \in \mathbb{R}^{21}$: 13 operational/circulation/HoldFB features + 8 causal directional demand-band features.
  - Action $a^U_m \in \mathcal{A}_U$ (`30` discrete joint plans: `5` loop headway factors $\times$ `3` inventory-guarded directional tilts $\times$ `2` turnaround budget retention tiers), projected onto remaining-day trips via exact quota-conserving `apply_global_quota_timetable`.
  - Critic/Actor: `IndexedDiscreteEnsembleQNetwork` ($K=6$, `hidden_dim=96`, LCB $\beta = -1.0$) + `CategoricalPlanPolicy` with bounded auto-entropy.
- **Lower Stop-Event Constrained SAC**:
  - State $s^L_e \in \mathbb{R}^{20}$: feature `0` is `limit_ratio = min(30, B_{v,t}) / 30` (`action_limit_feature_index=0`), plus 15 dimensionless headway/slack/load/progress features + 4 local demand-band features.
  - Action $u_e \in \{0, 5, 10, 15, 20, 30\}\text{ s}$, hard-masked by $u_e \le B_{v,t}$ across sampling, evaluation, policy optimization, and Bellman target backups.
  - Critic/Actor/Dual: `IndexedDiscreteEnsembleQNetwork` (`zero_hold_advantage=True`, $K=6$, `hidden_dim=96`) + `CostQNetwork` + `CategoricalPolicy` + Lagrangian multiplier $\lambda_L$ bounding zero-hold regularity regret and tight-slack holding excess.

## 4. Gate 2 Pass Criteria
1. **Control Gain**: At `ep200`, `FreqDuet-v7` improves `restricted_journey_s_per_pax`, `ewt_s`, and `plan_cv2` over `A1_fixed_bilateral` with 95% paired bootstrap CI strictly `< 0` on both `tidal_bimodal` and `mbta_route111_wkdy`.
2. **Long-Training Stability (`ep40 -> ep200`)**: No late-training degradation—mean `restricted_journey_s_per_pax` at `ep200` is at least as good as (or within `< 2.0 s/pax` non-inferiority margin of) `ep40` across seeds, with bounded readiness delay.
