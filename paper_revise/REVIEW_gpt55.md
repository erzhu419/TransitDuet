## SUMMARY

TransitDuet addresses an important and underexplored transit-control problem: joint timetable planning and real-time holding under asynchronous planning/control timescales. The high-level idea of TAP—propagating lower-level holding outcomes back to dispatch-level decisions—is potentially publishable for a transportation RL venue if made precise and empirically validated. However, the current paper substantially overclaims relative to its own tables: TransitDuet does **not** beat the strongest Fixed baseline on mean wait, CV, overshoot, or composite cost in Table 1; its only clear advantage is lower seed-to-seed variability. There are also serious internal inconsistencies in the action definition, TAP lifecycle mapping, constraint handling, stochasticity terminology, and ablation interpretation. In its current form, the paper is not yet convincing as a sound empirical contribution.

---

## STRENGTHS

- **Relevant transportation problem.** Joint timetable planning and holding control is a meaningful transit operations problem, and the asynchronous timescale issue is real.

- **Promising conceptual angle.** TAP as asynchronous cross-level feedback is a plausible and interesting extension of synchronous cross-advantage ideas such as RoboDuet.

- **Strong Fixed baseline is included.** The paper does not hide the fact that a hand-tuned fixed 360 s timetable is highly competitive, which is good scientific practice.

- **Some honest negative results.** Section 5.3 explicitly acknowledges that several ablations are within noise or even better on the in-distribution objective.

- **Operationally interpretable metrics.** Wait time, headway CV, fleet overshoot, and composite cost are appropriate transit-facing metrics.

---

## WEAKNESSES, RANKED BY SEVERITY

1. **The main empirical claim is not supported by Table 1.**  
   In Table 1, Fixed is better than TransitDuet on all four mean metrics: wait `7.03` vs `7.54`, CV `0.526` vs `0.561`, overshoot `2.17` vs `2.43`, and composite `1.62` vs `1.81`. The paper’s claim that TransitDuet “matches” Fixed is only defensible under a very loose “within seed std” criterion, and even then the learned method is uniformly worse on means. For a transit-RL reviewer, “higher mean cost but lower training-seed variance” is not sufficient unless operational-day variance is separately demonstrated.

2. **Variance framing is overstated and somewhat misleading.**  
   Section 5.2 argues that TransitDuet is “the deployable option” because its seed std is lower. But the reported std is across only **3 training seeds**, not necessarily across demand/traffic realizations under a fixed trained policy. In transit operations, robustness across days/demand scenarios is more important than retraining-seed variance. The paper needs evaluation variance over many stochastic rollout days, not just seed-to-seed training variability.

3. **Central mechanism TAP is not directly ablated.**  
   The paper’s main methodological contribution is TAP, but Table 2 does not include `--TAP`, `forward-only TAP`, `reverse-only TAP`, `β=0`, or different lifecycle aggregation choices. The ablations instead focus on HoldFB, CS-BAPR, Hindsight, MORL, Elastic, and DemandNoise, several of which are not clearly defined in Section 4. Without a no-TAP comparison, the paper does not show that TAP is responsible for the reported robustness.

4. **The definition of TAP’s “trip lifecycle” is internally inconsistent.**  
   Section 4.4 says `trip(i)` is the set of lower-level transitions “between dispatch event i and dispatch event i+1,” but also calls this the “complete trip lifecycle” of the trip initiated by dispatch `i`. These are not the same. A bus trip lifecycle should include that bus’s station arrivals until route completion, not all active-fleet station arrivals before the next dispatch. This matters because the claimed novelty is lifecycle aggregation.

5. **Upper action definition is inconsistent across the paper.**  
   Section 4.2 defines the upper action as three target headways `[H_peak, H_off, H_trans]`. Section 5.4 instead says the upper policy outputs a per-dispatch offset `δ_t ∈ [-120, 120]` relative to a baseline schedule. The introduction/results preview also refer to per-dispatch corrections. These are materially different action spaces, and the reader cannot tell what the upper policy actually controls.

6. **Constraint-handling claims are contradicted by the results.**  
   The abstract and Section 4.5 claim fleet-size constraints are enforced “without reward shaping,” but Eq. 15 explicitly modulates reward with a fleet-overshoot penalty, and Tables 1 and 3 show persistent nonzero overshoot. For example, TransitDuet has overshoot `2.43` in Table 1 and overshoot `3.0` for several fleet budgets in Table 3. This is not hard enforcement.

7. **The Lagrangian update appears mathematically wrong or at least incorrectly written.**  
   Eq. 8 updates  
   `log λ ← log λ + ηλ (c_limit - E[Qc])`.  
   For a standard constraint `E[c] ≤ c_limit`, λ should increase when `E[c] > c_limit`, i.e., with sign `E[Qc] - c_limit`. The current sign appears reversed. Also Algorithm 2 initializes `λ ← 0`, but Eq. 8 takes `log λ`, which is undefined.

8. **The OGD / measurement-projection section has sign and interpretation problems.**  
   Section 4.5 initializes `θ = [-0.5, -0.3, -0.2]` and defines reward as `R_U = θ^T p` with positive cost components. But Section 5.4 later says θ stabilizes around positive `[0.69, 0.23, 0.08]`; positive weights would reward higher wait/overshoot/CV if the policy maximizes reward. The paper needs a consistent cost-minimization vs reward-maximization convention.

9. **Generalization claims misuse “demand stochasticity” and “route stochasticity.”**  
   Section 5.6 evaluates travel-time/route stochasticity parameter `σ ∈ {0.5,1.0,1.5,2.0,3.0}`, but the abstract and results preview phrase this as demand stochasticity. Demand noise is separately defined as per-hour multiplier `N(1,0.15^2)`. These are not interchangeable. Also, the abstract says wait rises only 25% when route variance is doubled, but Table 5 shows `σ=3.0` versus training `σ=1.5` gives `9.41` vs `6.34`, a **48%** increase. The 25% increase corresponds to `σ=2.0`, not doubled `σ`.

10. **Several ablations undermine, rather than support, the proposed architecture.**  
   In Table 2, `--MORL`, `--Elastic`, and `--DemandNoise` have better mean composite cost than full TransitDuet. The paper says these help OOD or Pareto coverage, but the corresponding evidence is incomplete. Table 5 reports only the full model, not the ablations. The claim in the Figure 2 caption that the remaining ablations “fail on out-of-distribution generalization” is not supported by the provided table.

11. **Search baselines are weakly specified and likely unfair.**  
   GA and CMA-ES are given small populations and apparently noisy single-episode fitness. It is unclear whether they receive the same trained lower holding controller, the same evaluation budget, common random numbers, or repeated stochastic evaluations. Since Fixed beats both search methods by a large margin, the search baselines mostly show that the search setup is poor, not that TransitDuet is strong.

12. **Reference/writing quality needs tightening.**  
   Important acronyms and components—CS-BAPR, HoldFB, Hindsight credit, MORL scalarization, `G_no_demand_noise`—appear in experiments without being properly introduced in the method. The paper also says figures in the mechanism-analysis section are placeholders, which is not acceptable in a submission-ready manuscript.

---

## QUESTIONS FOR AUTHORS

1. **What exactly is the upper action?**  
   Is it `[H_peak, H_off, H_trans]`, a per-dispatch offset `δ_t`, or both? Please reconcile Section 4.2 with Section 5.4.

2. **What is the precise TAP assignment rule?**  
   Does a dispatch receive lower rewards from the bus trip it initiated, or from all lower-level events until the next dispatch? If the latter, why is this called a trip lifecycle?

3. **Where is the no-TAP ablation?**  
   Since TAP is the main contribution, please report full results for `β=0`, forward-only TAP, reverse-only TAP, and possibly lifecycle-sum vs lifecycle-mean aggregation.

4. **Are Fixed, GA, and CMA-ES using the same lower holding controller?**  
   If not, the comparison is confounded. If yes, was the lower controller trained jointly, frozen, or retrained per timetable?

5. **Why should seed-to-seed training variance be considered operational robustness?**  
   Please report variance over many stochastic evaluation episodes for each trained policy, separately from variance across random seeds.

6. **How is fleet feasibility enforced if overshoot remains nonzero?**  
   Tables 1 and 3 show persistent overshoot. Is fleet size a hard constraint, a soft penalty, or an elastic budget?

7. **Is the Lagrangian update sign correct?**  
   Eq. 8 appears to decrease λ when the expected cost exceeds the limit. Please derive the update and clarify how `λ=0` is handled in log-space.

8. **What evidence supports the claim that Hindsight, MORL, Elastic, and DemandNoise improve OOD behavior?**  
   Table 2 shows several of these ablations are better in-distribution. Please provide OOD tables comparing the full model against each ablation.

---

## INTERNAL CONSISTENCY CHECK

- **Table 1 contradicts the “dominates/matches Fixed” framing.**  
  TransitDuet has worse mean wait, CV, overshoot, and composite than Fixed. The paper should state this plainly.

- **CV bolding in Table 1 appears wrong.**  
  Lower is better, and Fixed has CV `0.526` while TransitDuet has `0.561`. TransitDuet should not be the only bolded CV entry.

- **“Variance” vs “standard deviation.”**  
  The paper says “3× lower seed variance,” but Table 1 supports 3× lower **standard deviation** for composite, not variance. The variance ratio would be about 9×.

- **β schedule inconsistency.**  
  Section 4.4 defines a linear ramp and explicitly says cosine schedules did not help. The abstract/discussion/table analogy refer to “cosine-annealed” coupling.

- **Action-space inconsistency.**  
  Section 4.2: upper action is `[H_peak, H_off, H_trans]`. Section 5.4: upper action is `δ_t ∈ [-120,120]`. These cannot both be the implemented action without further explanation.

- **Trip lifecycle inconsistency.**  
  Section 4.4 claims complete trip-lifecycle aggregation, but Eq. 9 defines aggregation over the interval between consecutive dispatches.

- **Fleet constraint inconsistency.**  
  The abstract says fleet constraints are enforced; Tables 1 and 3 show overshoot remains positive.

- **Demand vs route stochasticity inconsistency.**  
  Demand noise is `0.15`, while cross-σ generalization uses route stochasticity `σ=1.5`. The paper repeatedly blurs these.

- **Generalization percentage inconsistency.**  
  Doubling training `σ=1.5` to `σ=3.0` increases wait from `6.34` to `9.41`, about 48%, not 25%.

- **Table 1 vs Table 5 mismatch.**  
  Main results report TransitDuet wait `7.54`, while Table 5 reports `6.34` at the nominal training `σ=1.5`. If evaluation protocols differ, this must be explained.

- **θ sign inconsistency.**  
  Section 4.5 initializes negative θ weights for cost penalties, but Section 5.4 reports positive steady-state weights without explaining whether magnitudes or negated weights are plotted.

---

## RECOMMENDATION: **Reject**

The paper has a promising idea, but the current empirical evidence and internal consistency are insufficient: the strongest baseline beats TransitDuet on all mean metrics, the main TAP mechanism is not ablated, and several core definitions contradict each other.

---

## TOP 3 FIXES TO IMPROVE ACCEPTANCE ODDS

1. **Rebuild the empirical section around a fair, transit-relevant comparison.**  
   Add no-TAP, forward-only TAP, reverse-only TAP, and independent-training baselines. Evaluate all methods with many stochastic rollout episodes per trained seed. Report mean, standard error/confidence intervals, and operational-day variability, not only training-seed std.

2. **Fix the method specification.**  
   Decide whether the upper action is headways or per-dispatch offsets. Define TAP’s lifecycle mapping precisely. Correct the Lagrangian and θ-OGD sign conventions. Remove claims of hard constraint enforcement unless overshoot is actually zero.

3. **Reframe the contribution honestly.**  
   State that Fixed is the strongest mean-performance baseline and currently beats TransitDuet on means. If the contribution is robustness/adaptivity, prove it with appropriate OOD and demand-shift evaluations against Fixed and ablations, not just with lower training-seed variance.
