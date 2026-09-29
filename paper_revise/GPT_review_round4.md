# TransitDuet Round 4 Review

## 总体建议

**Weak Accept after cleanup / Minor-to-Major Revision。**

这一轮相比 round3 又前进了一大步。主文 lower 已经从旧的 DSAC/twin-Q 描述改成 RE-SAC ensemble + Lagrangian；Appendix 的 upper state/action 维度也从旧的 `5 -> 3` 改成了 `11 -> 1`；`generalization_eval.py` 改用 `runner_v3`，并补齐了 `sigma_route=1.5` 的 3 seeds × 10 eval；Pareto 也新增了 `eval_pareto_hiro.py`，从 validation-best `H_hiro` checkpoint 生成。也就是说，round3 中最硬的 paper-method-code 错配大多已经修掉。

现在主要风险集中在 **reproducibility pipeline 和几处残留不一致**。如果只看论文叙事，这版已经接近可投；如果审稿人会看 artifact 或要求从空日志目录复现表图，还会卡住。我的建议是：再做一轮小而硬的清理，尤其是 `run_paper_round3.sh`、baseline eval 参数、upper action squashing/alpha 描述，以及“统一 300 episodes”这类措辞。

## 这轮已经修好的点

1. **lower 方法描述基本对齐代码。**  
   `paper/sections/4_method.tex` 现在写的是 `RE-SAC--Lagrangian`，lower reward critic 是 `K=10` ensemble，actor loss 用 LCB，cost critic 单独建模，`lambda_lr=1e-4`，`alpha_max^L=0.1`。这和 `runner_v3.py + lower/resac_lagrangian.py + config_v2.yaml` 基本一致。

2. **Appendix 维度修正。**  
   Appendix upper architecture 现在是 11-dimensional upper state、scalar action `delta_t`，不再是旧的 5 维 state / 3 维 headway triple。

3. **旧 timetable / departure-offset 语义基本清掉。**  
   conclusion、keywords、problem statement 都已经转成 target-headway planning。PDF 中不再出现 `departure offset`、`headway triple` 这类旧主方法术语。

4. **θ-OGD 理论 claim 降级到合理范围。**  
   Section IV 现在明确说 OGD 的 `O(sqrt(T))` 是 OCO 动机，不声称非凸 SAC policy 有同样 feasibility guarantee。这是必要修正。

5. **Generalization 表有新脚本和新数据支撑。**  
   `logs/eval_generalization/cross_sigma/` 现在每个 `sigma ∈ {0.5,1.0,1.5,2.0,3.0}` 都有 3 seeds × 10 eval，聚合后能对上论文 Table V：
   `4.96 / 5.52 / 5.38 / 6.23 / 6.82` wait。

6. **Pareto 表有 H_hiro selected-checkpoint 数据支撑。**  
   `logs/H_hiro_seed*/pareto_frontier.json` 存在，3 seeds × 5 eval per fleet，聚合后能对上论文 Pareto 表。它不再明显来自旧 `A_full` 或 `H_tpc`。

7. **baseline 限制开始诚实披露。**  
   Discussion 现在明确说 search baselines 是 320 simulator episodes、GA/CMA-ES 用 final upper triple + validation-best lower checkpoint、rule baselines 是 1 seed sanity checks。这个比上一轮可信很多。

## 主要问题

### 1. `run_paper_round3.sh` 还不能真正“从空 logs 到论文表图”

脚本注释说它是 “from an empty logs/ to every table/figure”，但当前流程还没有闭合。

第一，Pareto 写入和作图读取路径不一致：

- `scripts/eval_pareto_hiro.py` 写到 `logs/H_hiro_seed*/pareto_frontier.json`;
- `scripts/make_result_figures.py` 读取的是 `logs_remote/H_hiro_seed*/pareto_frontier.json`;
- `run_paper_round3.sh` 的 `agg` 阶段只 mirror 了 `logs/eval_per_ckpt/*` 到 `logs_remote/eval_per_ckpt/*`，没有 mirror `logs/H_hiro_seed*/pareto_frontier.json` 或 training diagnostics。

因此从空目录运行 pipeline，`fig_pareto_frontier()` 可能直接 skip，training curves 也依赖 `logs_remote/H_hiro_seed*/diagnostics.csv`。当前本地能生成 figure，是因为 `logs_remote` 里已有手动/历史同步过的文件，不是因为脚本自洽。

第二，baseline eval 阶段没有传论文声明的 checkpoint/eval 参数：

```bash
python scripts/eval_fixed_baseline.py --seeds ...
python scripts/eval_baseline.py --method "$method" --seeds ...
```

但两个脚本默认不是论文 protocol：

- `eval_fixed_baseline.py` 默认 `--eps 49,99,119`，`--n_eval 60`;
- `eval_baseline.py` 默认 `--eps 49,99,119`，`--n_eval 20`;
- 论文 metrics 段写的是 `{49,99,149,199,249,299}` 且 20 eval episodes。

我用现有 JSON 对比了一下，如果只按脚本默认的 `49,99,119` 选 checkpoint，主表 baseline 会变成：

- Fixed composite 约 `1.849`，不是论文的 `1.544`;
- GA composite 约 `1.907`，不是 `1.800`;
- CMA-ES composite 约 `1.516`，不是 `1.488`。

也就是说，当前论文主表数值来自已有的更完整 JSON，而不是 `run_paper_round3.sh` 从空日志默认会复现出的结果。这个是当前最大的 artifact blocker。

建议：

- `eval_base` 阶段显式传 `--eps "$EVAL_EPS" --n_eval "$N_EVAL"`；
- `eval_fixed_baseline.py` 默认也改成 `49,99,149,199,249,299` 和 `n_eval=20`；
- 删除 `|| true`，至少在 full paper pipeline 中不要吞掉 baseline eval failure；
- `agg` 或 `figs` 前同步 `logs/H_hiro_seed*/diagnostics.csv/history.json/pareto_frontier.json` 到 `logs_remote/`，或者干脆让 `make_result_figures.py` 全部读 `logs/`。

### 2. 论文仍混用 “unified 300-episode protocol” 和 “near-unified 320-episode protocol”

Section V 的 baseline fairness 已经写清楚：TransitDuet 是 300 episodes，GA/CMA-ES 是 20 warmup + 300 search episodes，也就是 320 simulator episodes。但其他位置仍写得过强：

- `paper/sections/5_experiments.tex:53-55`: “Each method is trained for 300 episodes”；
- Table II 分组标题：`3 seeds, 300-ep from-scratch unified protocol`；
- Table II caption：`unified evaluation protocol` 且 “All methods ... validation-selected checkpoint”；
- Introduction / Conclusion 仍说 `unified 300-episode training and held-out evaluation protocol`。

这会被审稿人抓住，因为作者自己在 Discussion 承认它是 near-unified，不是 exact unified。建议全文统一成：

- main TransitDuet/ablations: 300 episodes;
- search baselines: 20 lower warmup + 300 upper-search episodes, 320 total;
- evaluation protocol统一，training budget approximately matched;
- GA/CMA-ES checkpoint selection是 final upper triple + selected lower checkpoint，不是严格 matched checkpoint pair。

表格 caption 尤其要改，不能一边说 unified 300，一边在 Discussion 才解释 320。

### 3. Upper policy 的 action squashing 和 entropy cap 仍不一致

现在出现两个新的小错配。

第一，主文和代码使用 sigmoid bounded Gaussian：

- `paper/sections/4_method.tex:50-56` 写 sigmoid 后 affine map 到 `[-120,120]`;
- `upper/resac_upper.py:53-121` 代码也是 sigmoid squash。

但 Appendix 写成了 tanh：

- `paper/sections/A_appendix.tex:131`: `tanh -> x 120`;
- `A_appendix.tex:137`: rollout 也是 `tanh x 120`。

这应改成 sigmoid + affine rescale，和主文/代码一致。

第二，upper entropy cap 不一致：

- `config_v2.yaml` 和 hyperparameter table 都是 `alpha_max^U = 0.05`;
- `runner_v3.py` 也传 `maximum_alpha=upper_cfg.get(..., 0.05)`;
- 但 `paper/sections/4_method.tex:75` 写 `alpha_max^U = 1.0`。

这个属于手写残留，建议直接改成 0.05。

顺手也可以修 `upper/resac_upper.py` 文件头，它还写 “Action dim=3: [H_peak, H_off, H_trans]”，虽然实际 runner 已经传 `action_dim=1` 和 `action_low/high=[-120,120]`。

### 4. Baseline table 的证据口径仍需更谨慎

论文现在承认了 GA/CMA-ES 的 residual differences，这是好事。但主表和正文仍有几处口径偏强：

- Table II caption 说 “All methods report the validation-selected checkpoint per seed”。对 GA/CMA-ES 更准确是 “validation-selected lower checkpoint paired with final upper triple”。
- `Because all four baselines share TransitDuet's lower-policy architecture and Lagrangian dual update, this gap isolates the effect of the goal-conditioned upper coupling` 仍略强。由于 search baseline 的 budget 多 6.7%、upper/lower checkpoint 不是严格成对、upper action family 是 static 3-headway triple，而 TransitDuet 是 contextual per-dispatch scalar shift，这个 gap 不能完全“isolate” goal-conditioned coupling，只能说 protocol controls for lower architecture to first order。
- Rule-lower 和 per-candidate rows 已经分组标注 1 seed，但正文 “against all six external baselines” 仍容易被理解成同等证据强度。建议主结论只基于 3-seed learned-lower rows，1-seed rows作为 sanity checks。

### 5. Official result artifacts 里仍混有旧文件，容易误读

本地 `logs/eval_generalization/demand_shift/` 同时有旧的 `G_in_dist_seed*.json` / `G_ood_noisy_seed*.json` 和新的 `demand_in_dist_seed*.json` / `demand_ood_noisy_seed*.json`。`scripts/aggregate.py`、`scripts/make_mechanism_figures.py` 也仍然写 `A_full`。这些未必影响当前 paper table，但 artifact reviewer 很容易误用。

建议把旧聚合脚本和旧 JSON 移到 `_archive_pre_round2`，或者在 `scripts/README` 中标注：

- paper pipeline: `run_paper_round3.sh`;
- deprecated: `aggregate.py`, old `A_full`, old `G_*` demand-shift JSON。

## 次要问题

1. `make_result_figures.py` 注释仍说 baselines train for 120 episodes，但现有 `upper_ga/upper_cmaes` histories 是 320 episodes，`upper_fixed` 是 300 episodes。

2. `paper/sections/5_experiments.tex:331` 说 evaluating on `{0.5,1.0,2.0,3.0}`，但 table 也包含 1.5。建议写 `{0.5,1.0,1.5,2.0,3.0}`。

3. Pareto section 说 “one trained policy” 容易和 “one policy per seed” 混淆。caption 已经写了 per seed，正文第一句也建议补 “per seed / validation-best checkpoint”。

4. Pareto frontier 的 `N=10` wait 最低但 overshoot=3，`N=14` composite 最低。现在解释已经比上一轮好，但摘要里只说 “produces a Pareto frontier” 就够了，不要暗示它天然给出单调 trade-off。

5. `generalization_eval.py` 文档说 “round-2 runner”，当前文件和 paper 都是 round3/round4语境，建议改成 “runner_v3” 即可。

## 建议的最低修复清单

1. 修 `run_paper_round3.sh`：
   - baseline eval 传 `--eps "$EVAL_EPS" --n_eval "$N_EVAL"`；
   - 不要 `|| true`；
   - mirror `logs/H_hiro_seed*/{diagnostics.csv,history.json,pareto_frontier.json}` 和 baseline histories 到 `logs_remote/`，或改 figure script 读 `logs/`。

2. 全文把 “unified 300-episode protocol” 改成 “near-unified / approximately budget-matched protocol”，并在 Table II caption 直接写清 baseline 是 320 total episodes。

3. 修 upper Appendix：
   - `tanh x 120` 改为 sigmoid + affine rescale；
   - `alpha_max^U=1.0` 改为 0.05；
   - `upper/resac_upper.py` 文件头 action dim=3 改成当前主用 action dim=1。

4. Table II caption 中明确 GA/CMA-ES 的 checkpoint pairing：final upper triple + selected lower checkpoint。

5. 清理或标注 deprecated scripts/files，避免 `A_full`、旧 `G_*` generalization、旧 aggregate 脚本被误认为当前 paper pipeline。

## 结论

这轮已经把前几轮最危险的科学错配基本修掉了。主方法现在讲得清楚，主结果、generalization、Pareto 都有当前 `H_hiro/runner_v3` 数据支撑；从论文质量看，已经接近可投。

我还不建议马上提交的原因是 artifact 仍有断点：`run_paper_round3.sh` 按当前默认参数复现不出 Table II 的 baseline 数字，figure script 还依赖 `logs_remote` 的手动同步状态，paper 中也仍有若干 “exact unified 300 episodes” 的过强措辞。把这些清掉后，这篇就可以进入正式投稿状态。
