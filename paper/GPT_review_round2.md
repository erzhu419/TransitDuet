# TransitDuet Round 2 Review

## 总体建议

**Major Revision / Weak Reject。**

这一版相比 round1 有明显进步：`δ_t` 的主线叙述基本收敛到“只改变 lower target headway、不改变实际 launch schedule”，Lagrangian 更新符号也修正了，`clim` 改成了 0.5，并且对 θ-OGD 的约束保证表述更谨慎。新增 rule-lower、per-candidate retrain baseline 也回应了上一轮关于 baseline 公平性的部分质疑。

但结合代码和实验脚本看，当前稿件仍有几个会直接影响可接收性的硬问题：论文描述的方法、主结果的复现入口、checkpoint evaluation 脚本和实际训练代码没有完全对齐。现在最主要的问题不是方法想法，而是“论文表格到底由哪套代码、哪套 runner、哪套 evaluation protocol 生成”还不够可信。

## 已改善的点

1. **核心 action 语义比 round1 清楚。**  
   主文现在明确说 `δ_t` 是 target-headway shift，launch time unchanged；`runner_v3.py` 的 HIRO mode 也确实在 `coupling_mode == "hiro"` 时设置 `trip._delta_t = 0`，并把 `trip.target_headway` 改为 `base + δ_t`。

2. **Lagrangian 符号修正。**  
   论文 Eq. 18 改成 `log λ + η(E[Qc] - clim)`，代码 `lower/resac_lagrangian.py` 里也用 `lambda_loss = -λ(cost - cost_limit)`，方向是对的。

3. **θ-OGD claim 降低了。**  
   Introduction 里已经写明它是 adaptive penalty / soft constraint，不再声称提供 worst-case feasibility guarantee。这是必要修正。

4. **generalization 命名更准确。**  
   Section V-F 改成 travel-time stochasticity，并明确区分 `σ_route` 和 demand noise，解决了上一轮的混淆。

5. **baseline 部分有回应。**  
   新增 rule-lower baseline 和 per-candidate-retrain GA，说明作者在尝试回应“搜索 baseline 是否不公平”的问题。

## 主要问题

### 1. 论文写 upper 是 REINFORCE，但代码实际是 RE-SAC

论文 Section IV-B 写 upper policy 使用 REINFORCE estimator，并给出 Eq. 12 的 score-function gradient。但代码里 `upper/resac_upper.py` 明确实现的是 `RESACUpperTrainer`：upper 有 replay buffer、ensemble Q、target Q、entropy temperature 和 SAC-style actor update。

相关代码证据：

- `transit_duet/upper/resac_upper.py`: `RESACUpperTrainer`
- `transit_duet/runner_v3.py`: upper update 调用 `self.upper_trainer.update(self.upper_batch_size)`

这不是实现细节差异，而是算法层面的差异。论文中的 upper-policy gradient、unbiasedness proposition、训练过程描述都不对应当前代码。必须二选一：

- 如果真实方法是 RE-SAC upper，就把论文里 REINFORCE 全部改成 off-policy RE-SAC upper，并说明 replay buffer、critic target、entropy、LCB/ensemble；
- 如果真实方法是 REINFORCE，就需要代码提供对应实现，并用它重新生成主实验。

当前状态下，审稿人无法判断 TransitDuet 的贡献到底来自 HIRO-style coupling，还是来自一个更强的 off-policy upper learner。

### 2. 主实验复现入口仍然跑错方法

`scripts/launcher.py` 的 Tier 1 main job 仍然运行：

```bash
python -u runner_v2.py --config configs_ablation/A_full.yaml ...
```

但论文主结果声称是 `TransitDuet (HIRO, Ours)`，对应配置应是 `configs_ablation/H_hiro.yaml`，且需要 `runner_v3.py` 才会识别 `coupling_mode: hiro`。`runner_v2.py` 不支持 HIRO goal-channel 分支，默认仍是 launch-time shift 版本。

这会导致一个严重复现问题：按仓库一键脚本跑出来的是 `A_full/channels`，不是论文 Table II 的 `H_hiro`。

建议：

- 把 paper results 的官方入口改成 `runner_v3.py --config configs_ablation/H_hiro.yaml`；
- 更新 launcher、README、aggregate scripts；
- 删除或明确标记旧的 `A_full` / `runner_v2` 结果，避免它们被误认为主结果。

### 3. Checkpoint evaluation 脚本可能用错 runner

论文 Table II caption 说主结果来自 validation-selected checkpoints。但 `scripts/per_ckpt_eval.py` 当前从 `runner_v2` import：

```python
from runner_v2 import TransitDuetV2Runner, load_config
```

如果用这个脚本评估 `H_hiro` checkpoint，`coupling_mode: hiro` 会被忽略，评估时会退回 launch-time shift 语义。这意味着训练时如果用 `runner_v3` 的 HIRO goal-channel，评估时却可能按 `runner_v2` 的 launch-shift 逻辑 rollout。这样 Table II 的 validation-selected 数字就不可信。

这是 round2 最关键的阻断问题之一。需要确认 Table II 的 `5.29 / 0.457 / 1.55 / 1.444` 到底由哪个脚本生成，并提供可复现命令。若确实由 `per_ckpt_eval.py` 生成，则必须改成导入 `runner_v3` 后重跑所有 selected-checkpoint evaluation。

### 4. 超参数表和代码配置仍不一致

论文 Table I / Appendix 写：

- upper lr = `1e-4`
- lower lr = `1e-5`
- lower hidden dim = 32
- batch size = 2048
- lambda lr = `1e-3`
- reward scale = 10

但 `transit_duet/config_v2.yaml` 写：

- upper lr = `3e-4`
- lower lr = `3e-4`
- lower hidden dim = 64
- lower batch size = 512
- lambda lr = `1e-4`
- reward scale = 1

如果论文表格是旧实验配置，应删除或更新；如果真实实验使用了另一份 config，应把对应 config 放进 repo，并在 paper 里引用 exact command/config path。现在的可复现性仍然不够。

### 5. “所有方法 300 episodes” 与实验脚本不完全一致

论文说 Fixed / GA / CMA-ES / TransitDuet 都是 300-episode unified protocol。但 `scripts/launcher.py` 对 baseline 使用：

```python
--episodes 100
```

本地日志里部分 `upper_ga/upper_cmaes` history 长度是 320，说明可能后来手动 resume 到 300+，但官方 launcher 和论文说法不一致。建议明确：

- 主表每个 row 的训练命令；
- 是否从 checkpoint resume；
- 每个 seed 的 selected checkpoint；
- evaluation seed set；
- 是否所有 methods 都同样做 validation checkpoint selection。

### 6. 主结果数值与 repo 中聚合结果不一致

`transit_duet/results_remote/summary.txt` / `results_remote/*.csv` 里的 baseline 和 ablation 数字与论文 Table II / III 不一致。例如 `results_remote` 里 `A_full` wait 约 7.5，baseline fixed wait 约 4.6；论文主表则是 TransitDuet wait 5.29、Fixed 5.60。可以理解为 `results_remote` 是旧版 `A_full` 结果，但仓库里没有清楚标记。

这会让审稿人或读者很容易拿到一组和论文不匹配的结果。建议把旧 results 移到 archive，保留唯一一套 paper-generated artifacts，或者在 `results/README.md` 写清楚每个结果文件对应哪版 paper。

### 7. “timetable planning” 的表述仍偏强

新版方法本质上是 adaptive target-headway setting：实际 baseline launch schedule 不变，upper 通过改变 lower 的目标 headway 让 holding layer 实现服务间隔调节。这比 round1 清楚了，但仍不完全等价于 timetable planning。

论文里仍有“timetable planner”“dispatch headway is the primary planning lever”“upper controls dispatch headway”等表述。Appendix 甚至写 “The upper policy controls the dispatch headway—the time gap between consecutive launches”，这和主文“launch schedule unchanged”不一致。

建议将贡献名从 “joint timetable planning and holding control” 降为更准确的：

- joint target-headway planning and holding control
- adaptive headway-target setting with holding control
- service-headway planning rather than timetable launch-time planning

如果坚持 timetable planning，需要说明“timetable”在本文中指 target service headway 而不是实际 departure timetable。

### 8. baseline 新增了，但论证方式仍有问题

Rule-lower baselines 是有用补充，但 Table II 把 3-seed SAC baselines、1-seed rule baselines、1-seed per-candidate baseline 放在一个表里，并用 “TransitDuet retains the leading composite against all six external baselines” 作结论，容易过度解释。

Per-candidate-retrain GA 的论证也不够强。论文正文说 6 generations × 6 candidates × (6 train + 2 eval)；代码默认是 8 pop × 5 gens × (30 train + 10 eval)，`eval_per_cand_baseline.py` 又加载固定默认 triple 和 checkpoint。当前无法确认 Table II 的 per-cand row 来自哪个设置。更重要的是，per-candidate baseline 只做 1 seed，且 fine-tuning budget 很短，用它证明 shared-lower protocol 是“appropriate comparison”结论过强。

建议把这些 baseline 移到 appendix 或 reviewer-response style discussion，不要作为强主表证据；主文只保留 seed 数和 protocol 一致的 baseline。

### 9. Composite cost 的计算口径还需明确

论文定义 composite 为：

```text
wait/10 + overshoot^2/Nfleet + CV
```

但部分脚本 `scripts/composite_score.py` 是先取平均 overshoot 后平方，即 `(mean overshoot)^2 / mean N`，而 evaluation 脚本是逐 episode 计算 composite 后再平均。两者在有方差时不同。论文表格看起来用的是逐 episode evaluation composite，但 aggregate/summary 工具不统一。

建议统一所有结果脚本：composite 必须逐 episode 计算，然后对 episode/seed 聚合；不要用 aggregate mean overshoot 后再平方。

## 次要问题

1. Section V 的 metrics 段说 “aggregated over final 30 training episodes”，Table II caption 又说 “validation-selected checkpoint per seed”。这两个协议不一样，应统一。若 Table II 是 validation-selected，metrics 段不要再说 final 30。

2. “TransitDuet improves on every metric” 仍不精确。Table II 中 wait 是 CMA-ES 5.24，TransitDuet 5.29；主文后来又说 tied on wait。建议标题改成 “matches or improves on the strongest baseline, with better composite”。

3. “seed variance 1.7-2.5× lower” 的文字仍引用了旧数值。新版 Table II 中 GA composite std 是 0.458，CMA-ES 是 0.022，TransitDuet 是 0.038。TransitDuet 并不是低于 CMA-ES。这里需要重新计算或删除。

4. Appendix 中 lower network 仍写 twin-Q / hidden 32，但代码是 RE-SAC ensemble Q / hidden 64。应整体重写 network architecture section。

5. `runner_v3.py` 的文件头、CLI description 仍写 `runner_v2.py` / `TransitDuet v2`，容易造成复现混乱。

6. HIRO mode 中 trip diagnostics 仍记录 `effective_launch = original + delta_t`，但实际 `_delta_t = 0`，这个字段会误导后处理脚本或图。

7. 论文里 “upper queried every 300s” 与实际环境 “per eligible trip dispatch callback” 之间仍需更精确对应。264 trips / day 与 M≈30-50 dispatch events 的描述也要统一。

## 建议的最低修复清单

1. **确定唯一 paper pipeline。**  
   写一个 `scripts/run_paper_round2.sh` 或 README block，能从空日志目录重现 Table II / III / figures。主方法必须用 `runner_v3.py + H_hiro.yaml`，或者论文改回实际 runner。

2. **修复 evaluation runner。**  
   `per_ckpt_eval.py` 必须按 experiment config 自动选择 `runner_v3`，或直接统一只保留一个 runner。

3. **重写算法描述以匹配代码。**  
   如果保留代码现状，upper 是 RE-SAC，不是 REINFORCE；lower 是 RE-SAC ensemble，不是 twin-Q SAC。相关公式、Algorithm 1、Appendix 全部要一致。

4. **统一超参数表。**  
   从实际 YAML 自动生成表，避免手写漂移。

5. **重新生成 Table II / III。**  
   用修复后的 runner/eval 脚本重新跑 selected checkpoint evaluation，并在 paper 中报告每个 seed 的 selected checkpoint。

6. **清理旧结果。**  
   `A_full`、`H_tpc`、`H_hiro`、`logs_remote`、`results_remote` 的关系要写清楚。否则外部读者无法判断哪个是 paper result。

7. **降低 timetable planning claim。**  
   除非实际改变 launch timetable，否则建议改成 adaptive target-headway planning。

## 结论

这版论文在叙事上比 round1 成熟了，很多明显的文字矛盾已经修掉。但代码审查暴露出更深的问题：**论文方法、主实验 runner、checkpoint evaluation 和超参数配置仍没有完全对齐**。在这些问题修复前，我不建议投正式会议/期刊，因为审稿人一旦要求代码复现，当前版本很容易被判定为实验协议不清或结果不可复现。

如果作者能统一 runner、重跑表格，并把 upper/lower 算法描述改成真实实现，这篇工作的核心想法仍然有投稿价值。
