# Paper Plan: TransitDuet

**Title**: TransitDuet: Bi-Level Reinforcement Learning with Temporal Advantage Propagation for Joint Bus Timetable Planning and Holding Control
**One-sentence contribution**: We propose TransitDuet, a bi-level RL framework that couples low-frequency timetable planning with high-frequency holding control through Temporal Advantage Propagation (TAP), enabling execution-layer feedback to reshape planning decisions while satisfying fleet-size and headway-uniformity constraints.
**Venue**: Transportation Research Part C (or IEEE T-ITS)
**Type**: Method paper
**Date**: 2026-04-06
**Page budget**: ~20 pages (journal format, double-column or single-column depending on venue)
**Section count**: 7

---

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Section |
|---|-------|----------|--------|---------|
| C1 | TAP enables execution feedback to planning: when lower-level struggles (bunching), upper-level learns to adjust headways | Ablation: Full vs no-TAP (β=0) on bunching rate, avg wait, reward curves | Needs experiment | §4, §5 |
| C2 | θ-OGD measurement projection keeps peak fleet ≤ N_fleet without manual weight tuning | Ablation: Full vs fixed-weight reward; plot θ weight evolution and peak_fleet over episodes | Needs experiment | §4, §5 |
| C3 | Lagrangian soft constraint improves headway uniformity vs unconstrained lower policy | Ablation: Full vs no-Lagrangian (cost_limit=∞); headway deviation distribution | Needs experiment | §4, §5 |
| C4 | Joint optimization outperforms decoupled (fixed timetable + RL holding) and flat (single-level) baselines | Main comparison: TransitDuet vs Fixed-TT, Flat-RL, no-TAP, no-θ-OGD, no-Lagrangian | Needs experiment | §5 |
| C5 | TransitDuet generalizes across traffic variability levels (σ) | Train σ=1.5, test σ∈{0.5,1.0,2.0,3.0}; report avg_wait, bunching_rate | Needs experiment | §5 |

---

## Structure

### §0 Abstract (~200 words)
- **What we achieve**: A bi-level RL framework (TransitDuet) that jointly optimizes bus timetable planning and real-time holding control, connected by Temporal Advantage Propagation.
- **Why it matters / is hard**: Timetable and holding control operate at fundamentally different timescales; existing methods treat them in isolation, so execution difficulties (e.g., bus bunching caused by unrealistic schedules) cannot feed back to improve plans.
- **How we do it**: Two independent policy networks coupled via TAP (cross-advantage injection across timescales), with measurement projection for fleet-size hard constraints and Lagrangian relaxation for headway-uniformity soft constraints.
- **Evidence**: Ablation study on 6 variants; generalization across traffic variability.
- **Most remarkable result**: [To be filled after experiments — e.g., "X% reduction in bunching rate and Y% reduction in passenger wait time compared to decoupled baselines"]

### §1 Introduction (2–2.5 pages)
- **Opening hook**: Urban bus systems face a fundamental coordination failure: timetables are designed offline assuming smooth operations, while real-time holding control must compensate for disruptions the timetable never anticipated.
- **Gap / challenge**: (1) Hierarchical RL methods (options, feudal) assume shared state spaces or synchronous execution — bus planning and control are asynchronous with different state/action spaces. (2) Existing bus RL work focuses on holding control alone with fixed timetables. (3) RoboDuet-style cross-advantage works only for synchronous co-execution.
- **One-sentence contribution**: see above.
- **Contributions**:
  1. We introduce Temporal Advantage Propagation (TAP), generalizing synchronous cross-advantage injection (RoboDuet) to asynchronous, multi-timescale hierarchical systems—the first such extension in either robotics or transportation RL.
  2. We propose an ApproPO-inspired measurement projection mechanism that adaptively modulates upper-level reward weights via online gradient descent, enforcing fleet-size constraints without manual tuning.
  3. We design a complete bi-level architecture (TransitDuet) with Lagrangian-constrained lower-level control that provably separates hard constraints (fleet) from soft constraints (headway uniformity).
  4. We demonstrate through comprehensive ablations that each component is necessary and that the joint system outperforms decoupled and flat baselines.
- **Results preview**: [Best quantitative result from experiments]
- **Hero figure**: Fig 1 — System architecture showing upper (timetable) and lower (holding) policies, the env, and three coupling mechanisms (TAP, θ-OGD, Lagrangian). Include arrows showing the feedback loop: upper → target_headway → env → lower rewards → TAP → upper advantage.
- **Key citations**: RoboDuet (2024), ApproPO (Miryoosefi et al., 2019), DSAC, bus holding control RL works, hierarchical RL surveys.

### §2 Related Work (2 pages)
- **Subtopic A: RL for Bus Holding Control** — single-level RL with fixed timetables (Wang & Sun 2020, Alesiani & Gkiotsalitis 2021, etc.). Gap: no feedback to planning.
- **Subtopic B: Bus Timetable Optimization** — mathematical programming, metaheuristics. Gap: static, ignores real-time execution.
- **Subtopic C: Hierarchical / Multi-Timescale RL** — options framework, feudal networks, HAM. Gap: assume synchronous or shared-state hierarchy.
- **Subtopic D: Cross-Layer Coupling in RL** — RoboDuet (cross-advantage), constrained RL (Lagrangian, ApproPO). Gap: RoboDuet is synchronous; ApproPO requires nested best-response.
- **Positioning**: TransitDuet is the first to combine asynchronous cross-advantage propagation with measurement-based constraint handling for a real transportation application.

### §3 Problem Formulation (1.5 pages)
- **Notation**: MDP tuple, upper/lower state/action spaces, timescale separation
- **Problem formulation**: Bi-level MDP with upper-level (timetable) MDP triggered at dispatch events and lower-level (holding) MDP triggered at station arrivals. Formally define the asynchrony.
- **Constraint specification**: Fleet-size hard constraint via measurement set C; headway-uniformity soft constraint via cost function.
- **Key equations**: Upper/lower MDPs, measurement vector z(π), constraint sets.

### §4 Method: TransitDuet (4–5 pages)
- **§4.1 Architecture Overview** — Two independent networks, event-driven interaction with env.
- **§4.2 Upper-Level Policy πU** — Gaussian policy, action space [H_peak, H_off, H_trans], state construction.
- **§4.3 Lower-Level Policy πL** — DSAC-Lagrangian, parameter-sharing single-agent architecture, cost = headway deviation², learnable λ.
- **§4.4 Temporal Advantage Propagation (TAP)** — Core contribution. Formal definition, comparison with RoboDuet's synchronous version, β annealing schedule. Include Algorithm box.
  - Eq: A_U^aug[i] = A_U[i] + β · mean(A_L[j ∈ trip_i])
  - Eq: A_L^aug[j] = A_L[j] + β · A_U[trip_of(j)]
- **§4.5 Measurement Projection for Fleet Constraint** — θ-OGD on polar cone, reward weight modulation.
- **§4.6 Training Procedure** — 3-stage training (warmup → ramp → full), Algorithm box for full training loop.

### §5 Experiments (4–5 pages)
- **§5.1 Experimental Setup** — Env description (264 trips, OD data, route data), hyperparameters table, baselines.
- **§5.2 Main Comparison** — TransitDuet vs 5 baselines (Table 1).
- **§5.3 Ablation Study** — TAP, θ-OGD, Lagrangian ablations (Table 2, training curves).
- **§5.4 Mechanism Analysis** — θ weight evolution, λ convergence, β effect, target_headway adaptation.
- **§5.5 Generalization** — Cross-σ robustness test.

### §6 Discussion (1 page)
- Why TAP works: intuition about feedback loops.
- Limitations: single route, simplified demand, no real-world deployment yet.
- Computational cost analysis.

### §7 Conclusion (0.5 pages)
- Restate contributions (not copy-pasted).
- Future work: multi-route extension, real-time deployment, demand uncertainty.

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Architecture diagram | TransitDuet system overview: upper/lower policies, env, TAP/θ-OGD/Lagrangian coupling | Manual (TikZ) | HIGH |
| Fig 2 | Line plot (3 panels) | Training curves: episode reward, bunching rate, avg wait — all 6 variants | Experiment logs | HIGH |
| Fig 3 | Line plot | θ-OGD weight evolution over episodes (3 lines: wait, fleet, bunching weights) | Experiment logs | MEDIUM |
| Fig 4 | Line plot | Lagrangian λ convergence over episodes | Experiment logs | MEDIUM |
| Fig 5 | Heatmap or bar | Target headway adaptation: learned H_peak, H_off, H_trans over episodes | Experiment logs | MEDIUM |
| Fig 6 | Bar chart | Generalization: performance across σ∈{0.5,1.0,1.5,2.0,3.0} | Experiment logs | MEDIUM |
| Table 1 | Comparison | Main results: 6 methods × 4 metrics (avg_wait, bunching_rate, peak_fleet, reward) | Experiment logs | HIGH |
| Table 2 | Ablation | Component ablation results | Experiment logs | HIGH |
| Table 3 | Hyperparameters | All hyperparameters | config.yaml | LOW |
| Algo 1 | Algorithm box | TAP: Temporal Advantage Propagation | Method description | HIGH |
| Algo 2 | Algorithm box | TransitDuet training procedure | Method description | HIGH |

**Hero Figure (Fig 1) Details**:
- Left: Upper-level policy (πU) box with inputs [hour, demand, fleet, prev_hw, unhealthy_rate] and outputs [H_peak, H_off, H_trans]
- Center: Bus environment with dispatch events (upper triggers) and station arrival events (lower triggers) on a timeline
- Right: Lower-level policy (πL) box with bus observations and holding time output
- Top arrows: TAP coupling — lower advantages aggregated per trip → upper advantage augmentation; upper advantage per trip → lower reward augmentation
- Bottom: θ-OGD measurement projection (episode-level feedback loop) and Lagrangian λ update
- Caption: "TransitDuet architecture. The upper-level timetable policy (πU) sets target headways at each dispatch event; the lower-level holding policy (πL) controls buses at station arrivals. Three coupling mechanisms connect the layers: Temporal Advantage Propagation (TAP) enables cross-timescale advantage injection, measurement projection (θ-OGD) adaptively weights upper-level objectives to enforce fleet constraints, and Lagrangian relaxation constrains headway deviation at the lower level."

---

## Citation Plan

- §1 Intro: RoboDuet [He et al. 2024], ApproPO [Miryoosefi et al. 2019], bus bunching survey, HRL survey
- §2 Related Work:
  - Bus holding RL: [Wang & Sun 2020], [Alesiani & Gkiotsalitis], [Chen et al.], DSAC [Duan et al. 2021]
  - Timetable optimization: [Gkiotsalitis & Cats 2021], [Ibarra-Rojas et al. 2015]
  - Hierarchical RL: [Sutton et al. 1999 options], [Vezhnevets et al. 2017 feudal], [Nachum et al. 2018 HIRO]
  - Constrained RL: [Tessler et al. 2019 RCPO], [Stooke et al. 2020], Lagrangian methods
  - Multi-timescale RL: [Lee et al. 2020], [Metelli et al. 2020 action persistence]
- §3 Problem: MDP definitions, bi-level optimization
- §4 Method: RoboDuet (cross-advantage), ApproPO (θ-OGD), SAC [Haarnoja et al. 2018], DSAC

*Note: All citations marked [VERIFY] need bibliographic verification before submission.*

---

## Next Steps
- [ ] /paper-write to draft LaTeX section by section
- [ ] Run experiments (300 episodes × 6 variants) to fill evidence
- [ ] /paper-figure to generate all figures from experiment results
- [ ] /paper-compile to build PDF
