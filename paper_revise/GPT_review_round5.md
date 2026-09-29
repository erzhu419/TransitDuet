# TransitDuet Round 5 Review

## 总体建议

**Weak Accept / Minor Revision。**

这轮相比 round4 已经把最危险的论文-代码错配基本修掉了：实验协议不再强行说成 exact unified 300 episodes，baseline eval 参数和一键脚本明显更接近论文口径，upper policy 的 `alpha_max^U`、action squashing、Appendix 架构也对齐了当前 `H_hiro / runner_v3` 主线。现在主要不是科学主张崩，而是 **少量可见文字残留 + artifact pipeline 还差最后闭环**。

我审的是当前 `paper/main.pdf`，PDF 时间为 2026-05-03 20:40，23 页。

## 这轮已修好的关键点

1. **训练协议口径明显更诚实。**  
   Abstract、Introduction、Section V、Table II caption 和 Discussion 都已经改成 “near-unified / approximately budget-matched”，并写清楚 TransitDuet 是 300 episodes，GA/CMA-ES 是 20 lower warmup + 300 search-evaluation episodes，也就是 320 simulator episodes。

2. **主表 caption 和 Discussion 已经披露 GA/CMA-ES 的软 checkpoint pairing。**  
   `paper/sections/5_experiments.tex:120` 现在明确说 GA/CMA-ES 是 “validation-best lower checkpoint paired with the search's final upper triple”，Discussion 也把它列为 residual difference。这个修正很重要，避免被审稿人认为在 checkpoint selection 上偷换。

3. **baseline eval 参数基本修对了。**  
   `transit_duet/scripts/run_paper_round3.sh:118-124` 已经给 `eval_fixed_baseline.py` / `eval_baseline.py` 显式传入 `--eps "$EVAL_EPS" --n_eval "$N_EVAL"`；两个 eval 脚本默认也已经是 `49,99,149,199,249,299` 和 `n_eval=20`。round4 里“脚本复现不出 Table II baseline 数字”的大问题基本解决。

4. **`run_paper_round3.sh` 的 mirror 步骤补了不少。**  
   `agg` 阶段现在会把 per-ckpt eval、H_hiro/variants 的 diagnostics/history/pareto、baseline histories、generalization JSONs 同步到 `logs_remote/`。这比上一轮只靠手动历史目录可靠很多。

5. **upper 方法描述和代码已对齐。**  
   `paper/sections/4_method.tex:75` 是 `alpha_max^U=0.05`，Appendix 也写成 sigmoid + affine rescale 到 `[-120,120]`；`upper/resac_upper.py` 文件头也说明主配置是 scalar `delta_t`，不是旧的 3-headway action。

6. **旧 artifact 入口被清理或标注。**  
   `scripts/aggregate.py` 已删除；新增 `transit_duet/scripts/README.md`，把旧 `A_full`、旧 generalization JSON、旧 aggregate flow 标为 deprecated。artifact reviewer 误用旧脚本的概率下降了。

7. **generalization 的 sigma 集合补齐。**  
   Section V 现在写 `sigma_route ∈ {0.5,1.0,1.5,2.0,3.0}`，包含 training value 1.5，和表格一致。

## 仍需修的主要问题

### 1. Fig. 2 caption 仍残留 “same 300-episode unified protocol”

`paper/sections/5_experiments.tex:167` 仍写：

```tex
Training curves of all four methods on the same 300-episode unified protocol
```

这和全文新口径冲突。当前论文自己已经承认 GA/CMA-ES 是 320 simulator episodes，且是 near-unified rather than exact unified。这个 caption 在 PDF 里很显眼，建议直接改成：

```tex
Training curves under the near-unified, approximately budget-matched protocol
(TransitDuet/Fixed: 300 episodes; GA/CMA-ES: 20 warmup + 300 search-evaluation episodes; 3-seed mean ± std, smoothed window 15).
```

如果 Fixed 也按当前 `run_upper_comparison.py` 从空 logs 训练成 320 episodes，那这里要统一写成 Fixed/GA/CMA-ES 的实际 budget；见问题 3。

### 2. 复现脚本还没有生成论文中的机制图

论文仍引用三张机制图：

- `figures/theta_evolution.pdf`：`paper/sections/5_experiments.tex:298`
- `figures/lambda_convergence.pdf`：`paper/sections/5_experiments.tex:309`
- `figures/delta_utilization.pdf`：`paper/sections/5_experiments.tex:322`

`transit_duet/scripts/README.md:20` 也说 `make_mechanism_figures.py` 属于 `figs` 阶段。但 `run_paper_round3.sh` 的 figs 阶段只调用：

```bash
python scripts/make_result_figures.py
```

见 `transit_duet/scripts/run_paper_round3.sh:197-199`。也就是说，一键脚本现在能生成 Fig. 2/3/4/7，但不会重新生成 theta/lambda/delta 三张机制图。当前 `paper/figures/` 里这三张图还在，所以 LaTeX 能编译；但从 “empty logs / fresh artifacts to every table/figure” 的 artifact claim 看，pipeline 还没闭合。

建议在 figs 阶段追加：

```bash
python scripts/make_mechanism_figures.py 2>&1 | tee logs/round3_mechanism_figures.log
```

或者把机制图函数合并进 `make_result_figures.py`，让单入口名副其实。

### 3. Fixed baseline 的 budget 口径仍有一点混乱

现在文本里有三种口径：

- `paper/sections/5_experiments.tex:31-32`：TransitDuet 300；search baselines GA/CMA-ES 320，Fixed 没被放进 320。
- `paper/sections/5_experiments.tex:53-54`：Fixed、GA、CMA-ES 都像是 20 warmup + 300 search-eval = 320。
- `paper/sections/5_experiments.tex:120` 和表格分组 `:126`：Fixed 被放进 “320-episode budget” 组里。

代码也有潜在分歧：`run_upper_comparison.py` 默认 `total_eps = lower_warmup + episodes`，launcher 对 `fixed` 也传 `--episodes 300`，所以从空 logs 重新跑会得到 320 episode 的 fixed training；但当前本地 `logs/upper_fixed_seed*/history.json` 长度是 300，`make_result_figures.py:53-57` 的注释也说 Fixed trains for 300 episodes。

这个不是主结果本身的致命问题，但 artifact reviewer 会问：Table II 的 Fixed 到底是 300 还是 320？建议二选一并全篇统一：

- 如果 Fixed 按 300：launcher 对 fixed 显式 `--lower_warmup 0`，表格不要把 Fixed 放进 320 group。
- 如果 Fixed 按 320：重新生成 fixed histories/eval，figure 注释和 caption 都改成 Fixed 320。

### 4. Overshoot 结果解释有自相矛盾

Table II 里 TransitDuet 的 overshoot 是 `1.60 ± 0.04`，CMA-ES 是 `1.63 ± 0.02`，lower is better，所以 TransitDuet 数值上也是最优。但当前文字仍有旧叙述残留：

- Abstract: “matches or improves ... except overshoot”
- `paper/sections/5_experiments.tex:144-145`: “best mean on three of four metrics ... second-best on overshoot”
- 同一段又写 “CMA-ES marginally edges out TransitDuet on overshoot ($1.63$ vs $1.60$ — a $-1.8\%$ advantage to TransitDuet)”，前半句和括号里的数值方向相反。
- Conclusion 也写 “best-mean status on three of four metrics”。

建议改成更稳的说法：

> TransitDuet has the best numerical mean on all four reported metrics, although the overshoot margin over CMA-ES is only 0.03 and should be treated as within seed noise; we therefore avoid claiming a substantive overshoot advantage.

这样既不夸大，也不会和表格数值打架。

### 5. `make_result_figures.py` 的 generalization 输入路径仍不统一

`make_result_figures.py` 大部分读 `logs_remote`，但 `EVAL = ROOT / 'logs' / 'eval_generalization'`。同时 `run_paper_round3.sh:188-192` 又把 generalization JSON mirror 到 `logs_remote/eval_generalization/`。完整 pipeline 先跑 genrl 再 figs 时不一定出错，但如果 reviewer 只拿 mirror 后的 `logs_remote` 运行 figures，或者执行 `--only figs`，generalization 图路径就容易和文档口径不一致。

建议改成：

```python
EVAL = LOGS / 'eval_generalization'
```

或者删除 generalization mirror，明确所有 generalization figures 都从 `logs/` 读。现在两边都存在，反而让复现边界不清楚。

## 次要问题

1. `make_result_figures.py:77` 注释仍写 baseline histories 是 “~120 eps each”，已经明显过时。

2. `runner_v3.py:6-8` 文件头仍说 round-2 critical path；现在 paper 和脚本都是 round3/round5 语境，建议改成 “current paper pipeline”。

3. `run_paper_round3.sh:10` 注释说 eval_main 是 “60 ckpts/exp”，实际是 6 checkpoints × 20 eval episodes，不是 60 checkpoints。建议改成 “6 ckpts/exp × 20 eval episodes”。

4. `launcher.py:189` 的 timeout 注释还写 “A_full can be slow”，属于旧命名残留。

5. Generalization table caption 写 “Mean ± std over 3 seeds × 10 evaluation episodes each”，但 Table V 只有 wait 列给了 `±`，CV 和 overshoot 没有 std。要么补齐 CV/overshoot std，要么 caption 改成 “Wait reports mean ± std; CV and overshoot report means”。

6. 主表里 3-seed rows 的 `±` 是 seed std，而 1-seed rule/per-cand rows 的 `±` 很可能是 episode-level std。表格分组已经标了 1 seed sanity checks，但 caption 最好再说明 “for 1-seed sanity rows, ± denotes held-out episode std rather than seed std”。

## 建议的最低修复清单

1. 修 Fig. 2 caption：删掉 “same 300-episode unified protocol”。

2. 在 `run_paper_round3.sh` 的 `figs` 阶段调用 `make_mechanism_figures.py`。

3. 明确 Fixed baseline 的训练 budget，并同步修改 launcher / table caption / figure comment。

4. 修 overshoot 叙述：不要再说 TransitDuet 是 overshoot second-best 或 “except overshoot”。

5. 统一 `make_result_figures.py` 的 generalization 读取路径。

6. 清理 stale comments：`~120 eps`、round-2 critical path、A_full timeout、60 ckpts/exp。

## 结论

这轮已经达到 “可以投前最后清理” 的状态。核心方法、主实验、Appendix、baseline eval 参数和旧 artifact 标注都比 round4 稳很多；如果审稿人主要看论文科学内容，我会倾向 Weak Accept。

我还不建议完全不改就提交，原因是现在剩下的问题都很容易被快速发现：Fig. 2 caption 还写 exact unified 300，overshoot 叙述和表格数值冲突，一键脚本没有生成机制图。把这些修掉后，这版的主要风险就只剩正常论文局限：3 seeds、single-route simulator、baseline family 覆盖有限。
