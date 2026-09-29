# TransitDuet Round 3 Review

## 总体建议

**Major Revision。**

这一轮比 round2 有实质进步：主叙事已经从“离线 timetable optimization / launch-time perturbation”基本收敛到“adaptive target-headway planning + holding control”；upper 的训练描述也改成了 off-policy RE-SAC；`launcher.py` 和 `per_ckpt_eval.py` 的主路径已经切到 `runner_v3.py + H_hiro.yaml`。主表中 `TransitDuet (HIRO)` 的 `5.11 / 0.434 / 1.60 / 1.418` 也能从 `logs/eval_per_ckpt/H_hiro/H_hiro_per_ckpt.csv` 中复核出来，说明 round2 最严重的主结果 runner/eval 错配已经修掉了一大块。

但这版还不适合直接投。剩余问题主要不是 idea，而是 paper-method-code-result 的闭环仍有几处硬漂移：lower 方法描述仍是旧的 twin-Q/DSAC 写法；Appendix 还保留 5 维 upper state / 3 维 action；generalization 和 Pareto 的脚本与结果很可能仍来自旧 `runner_v2/A_full` 或不完整评估；baseline 的“统一 300 episode / 相同 lower hyperparams / validation-selected checkpoint”说法和代码还不完全一致。审稿人如果看代码或要求复现，仍会质疑结果链条。

## 主要改进

1. **主实验入口基本修正。**  
   `transit_duet/scripts/launcher.py` 现在把主实验设为 `runner_v3.py --config configs_ablation/H_hiro.yaml`，而不是上一轮的 `runner_v2.py + A_full.yaml`。

2. **HIRO checkpoint evaluation 修正。**  
   `transit_duet/scripts/per_ckpt_eval.py` 现在从 `runner_v3` import，并显式说明不能用 `runner_v2` 评估 HIRO checkpoint。这解决了 round2 最关键的 evaluation semantics 错配。

3. **论文上层算法描述已从 REINFORCE 改为 RE-SAC。**  
   `paper/sections/4_method.tex` 中 upper 已描述 replay buffer、ensemble Q、LCB policy improvement 和 entropy tuning；Appendix 也补了一段说明“not as REINFORCE”。

4. **主表 TransitDuet 数字有可追溯 CSV。**  
   `H_hiro` 三个 seed 的 best checkpoint 分别约为：
   - seed 42: ep49, composite 1.397
   - seed 123: ep49, composite 1.436
   - seed 456: ep99, composite 1.421

   聚合后对应论文 Table II 的 `5.11 ± 0.17 / 0.434 ± 0.008 / 1.60 ± 0.04 / 1.418 ± 0.016`。

5. **Composite 计算口径修正。**  
   `scripts/composite_score.py` 已明确先逐 episode 计算 `wait/10 + overshoot^2/N_fleet + CV`，再聚合，避免上一轮的 Jensen mismatch。

6. **标题和问题定义更准确。**  
   标题已改成 joint bus target-headway planning and holding control；Problem section 也明确说 planning 是 closed-loop adaptive target-headway setting，不再强行等同 offline timetable optimization。

## 主要问题

### 1. 主文 lower 算法仍和代码不一致

主文 Section IV 还在用旧的 lower 描述：

- `paper/sections/4_method.tex:26` 写 lower 是 `DSAC-Lagrangian`;
- `paper/sections/4_method.tex:86-99` 写 lower policy / twin Q 是 hidden dim 32;
- `paper/sections/4_method.tex:98-108` 给的是 standard SAC twin-Q clipped-double-Q target;
- `paper/sections/4_method.tex:250-252` 仍说 lower cost 来自 twin Q-network 和 cost Q-network 的五个 backward passes。

但代码实际主路径是：

- `runner_v3.py` import `lower.resac_lagrangian.RESACLagrangianTrainer`;
- `config_v2.yaml` lower hidden dim 是 64;
- lower reward critic 是 `K=10` ensemble Q + LCB，不是 twin Q;
- cost critic 仍是单独 cost Q，但 reward side 已不是 clipped double-Q SAC。

Appendix 的 `Reward Q-ensemble` 段已经写对了，但主文 lower 小节还没同步。这个属于算法描述级别错误，不是措辞问题。建议把 lower 小节整体改成 RE-SAC-Lagrangian：policy net、reward Q ensemble、LCB actor loss、shared mean target、cost critic、lambda update 分开写；删除或降级 DSAC/twin-Q/hidden-32 的表述。

### 2. Appendix upper architecture 仍是旧维度

Appendix 与主文/代码冲突很明显：

- `paper/sections/A_appendix.tex:122` 写 “5-dimensional upper state”;
- `A_appendix.tex:129-132` 写 `Linear 5 -> 64`, `Mean head 64 -> 3`, `Log-std head 64 -> 3`;
- `A_appendix.tex:138` 还说 log-probabilities summed across 3 action dimensions。

但当前 `config_v2.yaml` 和 Problem section 都是：

- upper state dim = 11;
- upper action dim = 1;
- action 是 scalar `δ_t ∈ [-120, 120]`。

这会让读者以为论文主模型仍是旧的三段 headway triple policy。这个必须修，不然审稿人很容易判定实现和论文不一致。

### 3. 仍有 timetable / departure offset 旧语义残留

虽然主线已经改成 target-headway shift，但多处旧表述还会把问题拉回 launch-time timetable：

- `paper/sections/2_related_work.tex:16`: “maps aggregate state features to departure-time adjustments”;
- `paper/sections/3_problem.tex:15`: “dispatch headway ... primary planning lever”;
- `paper/sections/3_problem.tex:143`: “a single upper action (a headway triple)”;
- `paper/sections/A_appendix.tex:110`: “upper policy controls the dispatch headway -- the time gap between consecutive launches”;
- `paper/sections/7_conclusion.tex:5`: “per-dispatch departure offset δt”。

这些和 `paper/sections/4_method.tex:43-44` 的“scheduled launch time stays at baseline, target headway reset to `h_target + δ_t`”冲突。建议统一改成：

- target-headway shift;
- goal headway;
- closed-loop service-headway target;
- nominal timetable fixed, no launch-time perturbation in the main method。

关键词里的 `timetable optimization` 也建议改成 `target-headway planning` 或 `headway control`，否则摘要标题和 keywords 的定位仍不一致。

### 4. Generalization 结果链条不可信

论文 Section V-F 声称 generalization table 是 “3 seeds × 10 evaluation episodes each”，但本地结果和脚本不支持这个说法。

代码问题：

- `transit_duet/scripts/generalization_eval.py:32` 仍 import `runner_v2`;
- `generalization_eval.py:93-98` cross-sigma 仍加载 `logs/A_full_seed{seed}` 和 `configs_ablation/A_full.yaml`;
- `generalization_eval.py:123-128` demand-shift 仍加载 `G_no_demand_noise`，不是当前 `H_hiro_no_demand_noise`;
- `run_paper_round2.sh` 没有调用 `generalization_eval.py`，只在最后用 `make_result_figures.py` 读已有 JSON。

数据问题：

- `logs/eval_generalization/cross_sigma/` 中 `σ=0.5,1.0,2.0,3.0` 各有 3 个 seed;
- 但 `σ=1.5` 只有 `sigma_1.5_seed42.json`，而且 `n_eps=2`;
- 论文表 caption 却写 train `σ_route=1.5` 的行也是 3 seeds × 10 episodes。

因此 Table V 的训练分布行和 “trained H_hiro policy generalizes” 这两个说法都需要重跑或降级。最低修复是：用 `runner_v3 + H_hiro.yaml` 对 selected checkpoints 重新生成 `σ ∈ {0.5,1.0,1.5,2.0,3.0}`，每个 sigma 都 3 seeds × 10 eval episodes，并把脚本纳入唯一 paper pipeline。

### 5. Pareto frontier 仍可能来自旧 A_full / H_tpc，不是 H_hiro

论文 Table/Figure 的 Pareto 部分说 “one trained TransitDuet policy”，上下文默认是当前主方法 `H_hiro`。但脚本不是这样：

- `transit_duet/scripts/make_result_figures.py:161-166` 仍写死读取 `logs_remote/A_full_seed{s}/pareto_frontier.json`;
- 当前 `logs_remote/` 下没有 `A_full_seed*`，这些旧结果在 `_archive_pre_round2/logs_remote/A_full_seed*`;
- 当前 `logs_remote` 里能找到的 Pareto JSON 主要是 `H_tpc_seed*`，不是 `H_hiro_seed*`;
- `launcher.py` 虽然对 `H_hiro` 训练命令加了 `--eval_pareto`，但当前 `logs/H_hiro_seed42` 只有 checkpoint 目录，`logs_remote/H_hiro_seed*` 也主要是 diagnostics，没有看到 `H_hiro_seed*/pareto_frontier.json`。

所以论文 Pareto 表中的数值很可能不是当前主策略的 frontier，或者至少无法由当前脚本从当前日志复现。这个问题会直接影响 “single trained policy produces Pareto frontier across nine fleet budgets” 的贡献。建议用 validation-selected H_hiro checkpoint 明确重算 Pareto，而不是训练末尾顺手 eval 或读旧 A_full。

### 6. Baseline 的“统一协议”仍和代码不完全一致

论文写 “All four methods share the same lower-level RE-SAC Lagrangian controller, same elastic-fleet sampling, same 300-episode training budget”。但 baseline 代码仍有几处不一致：

1. `run_upper_comparison.py` 默认 `lower_warmup=20`，并设置 `total_eps = lower_warmup + episodes`。`launcher.py` 传 `--episodes 300` 时，baseline 实际是 320 simulated days，而不是论文说的 300。

2. baseline lower hyperparams 与 `config_v2.yaml` 不完全一致。`run_upper_comparison.py:204-209` 使用 `lambda_lr=1e-3`、`maximum_alpha=0.3`；主 `H_hiro` config 是 `lambda_lr=1e-4`、`maximum_alpha=0.1`。`eval_baseline.py` 和 `eval_fixed_baseline.py` 也用 `lambda_lr=1e-3`、`maximum_alpha=0.3`。这削弱了 “same lower controller / same hyperparameters” 的 claim。

3. `eval_baseline.py:101-106` 对 GA/CMA-ES 读取的是 `history['upper_params'][-1]`，然后用这个 final upper triple 去评估所有 lower checkpoints。也就是说如果 best lower 是 ep99，upper 仍可能是 ep319 的最终搜索结果。这不是严格的 checkpoint-pair evaluation，会引入未来信息。

4. `run_paper_round2.sh` 只自动跑 `H_hiro/H_tpc/H_haar` 和 HIRO ablation 的 `per_ckpt_eval.py`，没有把 fixed/GA/CMA-ES 的 `eval_baseline.py` 纳入官方 pipeline。虽然本地 `logs/eval_per_ckpt/{fixed,ga,cmaes}` 的 JSON 能复核主表数值，但一键复现脚本没有覆盖它们。

建议把 baseline protocol 明确成二选一：

- 真正统一：300 episode 包含 warmup，所有方法同一 lower hyperparams，评估 checkpoint 时 upper/lower 使用同一 episode 的状态；
- 或承认 baseline 是近似统一协议，并在表 caption/limitations 中说明差异。

### 7. θ-OGD 的理论保证仍写得偏强

Abstract 和 Discussion 已经把 θ-OGD 降级成 soft adaptive penalty，这是正确方向。但 `paper/sections/4_method.tex:199` 仍写：

> guarantees that the cumulative constraint violation grows sublinearly, yielding an O(sqrt(T)) regret bound

这里不够严谨。当前实现不是 ApproPO 的 occupancy-measure projection，也没有 convex best-response oracle；只是用测量值更新 reward weights，然后让非凸 SAC 在 replay buffer 上近似响应。可以说 OGD 本身在 convex OCO 上有 regret bound，但不能直接推出当前 learned policy 的 cumulative constraint violation bound。建议改成：

> The `1/sqrt(t)` schedule is motivated by standard OCO regret analysis, but in our non-convex SAC setting we treat it as a stabilizing adaptive penalty rather than a formal feasibility guarantee.

这样和摘要的谨慎表述一致。

### 8. Generalization/figure pipeline 与 paper pipeline 没闭合

`run_paper_round2.sh` 的注释说 final outputs 包括 `generalization.pdf`，但脚本没有生成 generalization JSON；`make_result_figures.py` 只是读已有文件。Pareto 也类似，figure script 读 `A_full_seed*`。这意味着从空日志目录运行官方脚本，未必能复现论文所有表图。

建议新增一个真正的 `run_paper_round3.sh` 或更新现有脚本，至少包含：

- train main/baselines/ablations;
- eval selected checkpoints for all main table rows;
- eval Pareto from selected H_hiro checkpoints;
- eval generalization from selected H_hiro checkpoints;
- aggregate all table values into machine-readable CSV;
- generate figures only从这些 CSV/JSON 读数。

## 次要问题

1. `runner_v3.py` 文件头已经清楚很多，但 CLI print 和 argparse description 仍写 `TransitDuet v2`。这不是科学问题，但会继续制造复现混淆。

2. `scripts/make_result_figures.py` 顶部注释仍说 training curves 是 “A_full vs baselines”，颜色 key 也还叫 `A_full`。建议统一改成 `H_hiro`。

3. Section V-F 的 “generalizes symmetrically” 表述偏强。`σ=3.0` wait +48%，`σ=2.0` +25%，低 sigma 端则接近训练分布；这不是严格对称，只能说没有崩溃。

4. Table II 把 3-seed learned baselines、1-seed rule baselines、1-seed per-candidate GA 放在同一主表里。现在 caption 已经分组说明，比上一轮好，但正文仍不宜用它们支撑强结论。最强结论应主要基于 3-seed Fixed/GA/CMA-ES/SAC-lower rows。

5. Per-candidate GA 的结果很差，但只做 1 seed 且每 candidate fine-tune 很短。建议保留为 sanity check，不要用来证明 per-candidate retraining 一般不可行。

6. `H_hiro_no_tpc` 在 selected-checkpoint CSV 中有某些 seed 的 best composite 甚至优于 full seed 对应结果；论文已说 TPC 的 in-distribution effect 小，但 “TPC prevents covariate shift” 的主张最好更多依赖 launch-shift coupling 或 OOD evidence，而不是当前 in-distribution table。

## 建议的最低修复清单

1. **重写主文 lower 小节。**  
   删除 DSAC/twin-Q/hidden-32 描述，改成 `RESAC-Lagrangian`：policy hidden 64、reward Q ensemble K=10、LCB actor loss、cost critic、lambda update。

2. **修 Appendix architecture。**  
   upper 改成 state dim 11、action dim 1；lower 与 config/code 保持一致。

3. **清理所有旧 action/timetable 语义。**  
   全文统一为 target-headway shift；避免 departure offset、departure-time adjustment、headway triple、controls dispatch headway。

4. **重跑 generalization。**  
   用 `runner_v3 + H_hiro` selected checkpoints，所有 sigma 都 3 seeds × 10 eval episodes，尤其补齐 `σ=1.5`。

5. **重跑 Pareto。**  
   从 H_hiro selected checkpoint 生成 `H_hiro_seed*/pareto_frontier.json`，并让 figure/table 脚本只读这些文件。

6. **修 baseline protocol。**  
   统一 lower hyperparams、episode budget、checkpoint-pair evaluation；或者在论文里明确说明差异，不再声称完全相同。

7. **做一个唯一复现入口。**  
   从空 `logs/` 到 paper tables/figures 可一键复现；旧 `A_full/runner_v2` 结果只留 archive，并在 README 中标注不属于当前主结果。

## 结论

这轮已经从 “runner/eval 明显错配” 前进到 “主结果大体可追溯，但若干表图和文字仍有旧 pipeline 残留”。核心 idea 现在更清楚，`H_hiro` 主表也比 round2 可信得多；如果只看方法方向，已经接近一篇可投的工程型 RL/transit paper。

但在正式投稿前，必须把 lower 算法描述、Appendix 维度、generalization/Pareto 数据来源、baseline 统一协议这四块补齐。否则审稿人一旦查代码，仍会认为 paper claim 和 released artifacts 没有完全对齐。我的建议是继续大修一轮，而不是现在提交。
