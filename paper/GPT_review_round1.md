总体建议：Major Revision / Weak Reject

论文选题有价值：把公交时刻表规划和实时 holding 控制放进双层 RL 框架，问题动机清楚，仿真场景也比较贴近实际运营。TransitDuet 的目标条件化 upper-lower coupling、elastic fleet 训练、holding feedback 和 hindsight credit 都是合理方向。但目前稿件存在若干核心定义不一致和实验可信度问题，已经影响主贡献是否成立，建议大修后再投。

主要优点

问题重要，公交 timetable 和 holding control 的耦合确实是现有工作常常分开处理的缺口。
双时间尺度、异步 upper/lower 决策的动机讲得比较清楚。
消融实验覆盖了多个组件，至少能看出作者尝试解释机制，而不是只报最终分数。
Pareto frontier across fleet budgets 是一个有实际运营意义的展示方式。
主要问题

核心 action 定义前后矛盾。
文中有时说 δt 改变实际发车时间 tbase + δt，有时又说 launch schedule 不变，只改变 lower 的目标 headway htarget + δt。如果 launch 不变，那么方法更像“自适应 headway target setter”，不能严格称为 timetable planning；如果 launch 改变，那么 goal-conditioned coupling 和 TPC 的分布偏移论证要重写。这个问题必须首先澄清。

多个实现细节不一致，影响可复现性。
upper action 有时是 scalar，有时 appendix 网络输出是 64 -> 3；upper state 主文写 5 维，但 HoldFB 又加入 4 维，ablation 还说 zero state dims [5:8]；lower action 主文是 [0,60]，appendix 又是 [-60,60] 并解释负值为 early dispatch。hindsight credit 也有两套公式：Eq. 21/32 用归一化 gap deviation 且 αH=0.5，Section IV-G Eq. 27 又写 -|gi-htarget| 且 αH=0.1。

Lagrangian 更新符号疑似错误。
Eq. 18 写的是 λ += η(clim - E[Qc])。对于约束 E[c] <= clim，当 cost 超标时 E[Qc] > clim，λ 应该增大，而这个公式会减小 λ。文中后面又说 violation 时 λ rises，和公式相反。此外 clim 在表中是 0.15，机制分析里又写 0.5。

“θ-OGD projection enforces constraint without reward shaping” 说法过强。
实际写法是用 θ 调整 upper reward 中 wait/overshoot/CV 的权重，本质上仍是 adaptive penalty/scalarization，不是严格的 policy projection，也没有 ApproPO 那种 oracle 或可行性保证。建议把“enforces”改成“encourages / adaptively penalizes”，并删弱理论保证表述。

实验结论偏强。
只有 3 个 seeds，且很多指标误差范围重叠。例如 wait：TransitDuet 5.29 ± 0.29，GA 5.33 ± 0.26，差距很小；CV 与 CMA-ES 也类似。建议至少 10 seeds，报告置信区间或 paired test。现在“best on every metric”的文字可以保留，但不能据此声称显著优于所有 baseline。

baseline 公平性需要重写。
GA/CMA-ES 是如何和 lower SAC 共同训练的？每个 candidate 是否重新训练 lower？还是共享一个 lower？300 episodes 对 population search 是否公平？此外缺少更强的 timetable baseline，例如按时段优化 headway、MPC/rolling horizon、传统 holding rule + optimized timetable、只训练 upper/只训练 lower 的对照。

指标和结果有若干自相矛盾。
composite cost 定义为 wait/10 + overshoot^2/Nfleet + CV，但 Table II 数值看起来对不上。Table II 说 validation-selected checkpoint，前文又说 final 30 training episodes。Figure 6 文中说 λ 收敛到 0.57 ± 0.16，图里标注 converged λ = 1.89。这些会严重削弱审稿人信任。

generalization 表述混淆。
Section V-F 说 “demand stochasticity”，但表里 σ 实际是 route/travel-time stochasticity；同时训练的 demand noise 是 0.15。建议明确区分 passenger demand noise、travel-time noise、route stochasticity。

建议修改方向

先统一 TransitDuet 的真实控制语义：δt 到底控制实际发车，还是只控制 lower target。
全文统一 state/action 维度、action range、hindsight credit、clim、dispatch event 数量。
修正 Lagrangian 符号，并重新跑或至少核对相关实验。
降低理论 claim，尤其是 θ-OGD constraint guarantee。
增加 seeds、显著性检验和更强 baseline。
补充 simulator calibration：真实数据来源、OD/速度拟合误差、是否与实际 wait/headway 分布对齐。
给出代码、配置和 checkpoint selection 协议，否则 20 页里细节很多但仍难复现。
一句话评价：想法是有潜力的，但当前稿件最大问题不是实验分数不够高，而是核心方法定义和训练协议不够自洽。先把一致性和可复现性修好，再谈贡献强度。