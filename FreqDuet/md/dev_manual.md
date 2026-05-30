我：

给个完整的施工方案 确保上层是时刻表，主要根据低频信息决策生成；下层是动态控制，主要根据高频信息决策生成 上层不一定还要用haar或者EMA 选最合适的就行，当然也不是说一定要上下层完全独立（因果性应该就是为了保证这点），目的是为了更高效，更专注，上层专注在低频的深层规律，下层专注在高频的快速变化，不是说不可以相互转化（有点non-stationary那意思了)

GPT：

可以。我的建议是把下一版不要叫“给 TransitDuet 加小波特征”，而是明确做成：

# FreqTransitDuet：面向外生时间序列的频率分工式 HRL

核心目标是：

[
\text{上层}=\text{低频需求驱动的时刻表 / headway planning}
]

[
\text{下层}=\text{高频扰动驱动的站点级 holding control}
]

但不是把上下层完全切死。更合理的机制是：**低频主要给上层，高频主要给下层；当高频冲击持续存在，它会被“升格”为新的低频状态；当低频计划无法执行，下层反馈会回流给上层修正计划。**

这正好接在你现有 TransitDuet 上。现在 paper 里已经是上下层异步：上层在 dispatch event 做 per-dispatch target-headway shift，低层在每个 station arrival 做 holding；二者不同状态空间、不同时间尺度、无共享 clock。 现有 HIRO-style coupling 也是上层输出 (\delta_t)，下层只把 (h^{target}+\delta_t) 当作目标 headway，不把上层 advantage 注入低层 transition，这个结构应该保留。 你的 repo 里 `runner_v3.py` 也已经把 `hiro | haar | channels` 三种 coupling mode 分开，HIRO 模式明确是“上层 (\delta_t) 改 target headway，launch time 不变”，这是最适合改成频率分工版的入口。([GitHub][1])

---

## 1. 总体架构：不要先选 Haar/EMA，而是先定义“频率分工接口”

我建议整体结构改成五个模块：

[
\text{Demand Stream}
\rightarrow
\text{Causal Frequency Decomposer}
\rightarrow
\begin{cases}
\text{Low-frequency Planner / Timetable Upper}\
\text{High-frequency Holding Lower}
\end{cases}
\rightarrow
\text{Coupled Transit Environment}
]

名字可以叫 **F-HIRO TransitDuet** 或 **FreqTransitDuet**。

具体模块如下：

| 模块                                | 作用                                                          | 主要服务谁 |
| --------------------------------- | ----------------------------------------------------------- | ----- |
| Demand Intensity Estimator        | 把 passenger arrival count / OD count 估计成连续需求强度 (\lambda(t)) | 上下层共享 |
| Causal Frequency Decomposer       | 在线分解成 (\lambda^L, \lambda^M, \lambda^H)                     | 分工核心  |
| Low-frequency Timetable Planner   | 根据 (\lambda^L) 生成未来一段时间的 target timetable/headway plan      | 上层    |
| High-frequency Holding Controller | 根据 (\lambda^H)、局部 headway、load、speed shock 做 holding        | 下层    |
| Cross-frequency Promotion Gate    | 判断高频冲击是否持续，是否应该升格为低频 regime change                          | 上下层连接 |

重点是：**频率分离不只是 feature engineering，而是定义 HRL 的职责边界。**

---

## 2. 需求分解方法：首选 Causal Structural Time-Series，不首选 Haar/EMA

你的公交需求有几个特点：明显早晚高峰、OD 矩阵、Poisson 到达、日内周期、局部 burst、在线部署不能看未来。paper 里环境已经是 time-varying OD matrices，乘客每秒按 (\lambda_{od}/3600) 的 Poisson 过程生成，waiting time 从出现到上车计算。

所以我建议第一版不要用 Haar，也不要只用 EMA。最合适的是：

## Causal Dynamic Harmonic State-Space Model

也就是 **动态傅里叶季节项 + 局部趋势 + Poisson / Negative-Binomial observation + Kalman / particle filter 在线更新**。

形式如下：

[
y_{g,t} \sim \text{NegBin}(\lambda_{g,t}, \phi)
]

[
\log \lambda_{g,t}
==================

\mu_{g,t}
+
\sum_{k=1}^{K}
a_{g,k,t}\sin(2\pi k t / T)
+
b_{g,k,t}\cos(2\pi k t / T)
+
\beta^\top x_t
+
\epsilon_{g,t}
]

其中 (g) 可以是 route-direction、station-direction、或者 low-rank OD factor。(T) 是一天的周期，(K=3\sim 6) 足够表达早高峰、平峰、晚高峰。动态参数按随机游走更新：

[
a_{g,k,t}=a_{g,k,t-1}+\eta^a_{g,k,t}
]

[
b_{g,k,t}=b_{g,k,t-1}+\eta^b_{g,k,t}
]

低频需求定义为模型的 trend + smooth seasonal forecast：

[
\lambda^L_{g,t}=\exp
\left(
\mu_{g,t}
+
\sum_{k \le K_L}
a_{g,k,t}\sin(2\pi k t/T)
+
b_{g,k,t}\cos(2\pi k t/T)
\right)
]

高频需求定义为 causal innovation：

[
r_{g,t}
=======

\frac{y_{g,t}-\hat{\lambda}^L_{g,t}}
{\sqrt{\hat{\lambda}^L_{g,t}+\epsilon}}
]

[
\lambda^H_{g,t}
===============

\text{HPF}*{causal}(r*{g,t})
]

这个选择比 Haar/EMA 更适合你的项目，原因是：

第一，它是 causal 的，只用 (t) 以前的信息。第二，它天然给上层 forecast horizon，比如未来 30–90 分钟低频需求。第三，它有 uncertainty，可以把“低频不确定性”作为上层 state。第四，残差 innovation 自然就是下层要看的高频冲击。第五，高频残差如果持续偏正，可以通过 state-space 的 process noise 或 changepoint gate 自动变成新的低频趋势。

如果你想先快速落地，可以先用简化版：

[
\lambda^L_t = \text{causal spline / dynamic Fourier forecast}
]

[
\lambda^H_t = y_t-\lambda^L_t
]

等主实验跑通后，再把 estimator 换成 learnable 或 Bayesian 版本。

---

## 3. 频段划分：按 TransitDuet 的控制周期来定，而不是拍脑袋

你当前系统里，上层大约每个 dispatch event 触发，低层在每个站点到达触发，paper 里也强调两层不是同步时钟，且一个上层 action 期间会有大量低层动作。 repo 的 `runner_v3.py` 也写明 HIRO 模式是一日约 264 个 dispatch event，而不是固定 300 秒 timer。([GitHub][1])

所以频段建议这样定义：

[
\lambda(t)=\lambda^L(t)+\lambda^M(t)+\lambda^H(t)
]

其中：

| 频段             |      时间尺度 | 给谁                     | 含义                     |
| -------------- | --------: | ---------------------- | ---------------------- |
| 低频 (\lambda^L) | (>30) min | 上层主输入                  | 早晚高峰、平峰、长期供需趋势         |
| 中频 (\lambda^M) |  5–30 min | 上层/下层共享的 regime buffer | 持续拥堵、持续客流偏移、事件开始       |
| 高频 (\lambda^H) |  (<5) min | 下层主输入                  | 局部客流 burst、某站突然积压、速度扰动 |

这里的 30 分钟不是绝对值，而是建议起点。更严格可以用数据自动选 cutoff：

[
T_c^U
=====

\max(4\bar{H}_{dispatch}, 20\text{ min})
]

[
T_c^L
=====

\min(2\bar{T}_{station}, 5\text{ min})
]

上层看 (T_c^U) 以上的慢变化，下层看 (T_c^L) 以下的快变化，中间频段用于判断“高频是否正在变成新的低频”。

---

## 4. 上层必须从“每次 (\delta_t)”升级为“时刻表生成器”

现在 TransitDuet 的上层 action 是 1D (\delta_t)，`config_v2.yaml` 里也是 `action_dim: 1`、`delta_max: 120.0`，环境有 `demand_noise: 0.15`、`effective_trip_num: 264` 等配置。([GitHub][2]) 这对于证明 bi-level coupling 可以，但如果你下一篇要强调“上层是时刻表”，最好不要让上层看起来只是每车一个 target shift。

我建议改成 **receding-horizon timetable generator**：

[
\pi_U:
s^U_t
\mapsto
\mathbf{c}_t
]

[
H_U(\tau, d)
============

H_0(\tau,d)
+
\sum_{m=1}^{M_c} c_{m,d} B_m(\tau)
]

其中 (B_m(\tau)) 是 cubic B-spline basis，(H_U(\tau,d)) 是未来 (T_U^{horizon}) 内某方向 (d) 的 planned headway curve。

然后按 headway curve 生成 dispatch timetable：

[
T^{sched}_{i+1,d}
=================

T^{sched}*{i,d}
+
H_U(T^{sched}*{i,d}, d)
]

这样上层 action 不再是“每次抖一下 (\delta_t)”，而是输出未来 30–60 分钟的 **平滑 headway plan**。这就是时刻表。

### 上层 action 建议

第一版可以设：

[
a_U=[c_{1,up},c_{2,up},c_{3,up},c_{1,down},c_{2,down},c_{3,down}]
]

即每个方向 3 个 spline coefficient，覆盖未来 45 分钟。每 15 分钟 replanning 一次，但只执行前 10–15 分钟，类似 rolling horizon。

为了防止上层高频乱动，加低通约束：

[
\mathcal{L}_{U,smooth}
======================

|\Delta^2 H_U|^2
]

[
\mathcal{L}_{U,HF}
==================

|\text{HPF}(H_U)|^2
]

这会强制上层只做低频 planning。

### 两个实现版本

**MVP 版本：target timetable，不真实改 launch time。**

保留你现有 HIRO mode：launch time 不变，trip 的 target headway 改成 (H_U(t))。这和当前代码最兼容，因为 `runner_v3.py` 现在已经在 HIRO mode 下把 `trip.target_headway = base_hw + delta_t`，且 launch shift 设为 0。([GitHub][1])

**完整版：terminal dispatch timetable，真实首站 holding。**

把上层输出的 (H_U(t)) 用于 terminal dispatch：

[
launch_i^{actual}
=================

\max(ready_i, T_i^{sched})
]

也就是上层真的决定首站发车时刻；下层只在中间站做 holding。这更符合你说的“上层时刻表本质是首站 holding”。这一步要改环境 dispatch scheduler，但论文叙事更干净。

我建议工程上先做 MVP，实验确认有效后再做完整版。

---

## 5. 上层 state：低频为主，高频只给摘要，不给原始高频

新的上层 state 可以从当前 11 维扩展到 25–40 维。当前 paper 的 upper state 包括 hour、demand、fleet、prev_headway、holding_ratio、HoldFB、target headway、direction、fleet budget。 新版要把原来的 `demand/1000` 替换成频率分解后的低频需求状态。

建议：

[
s^U_t=
[
time,
dir,
N_{fleet},
fleet_on_route,
prev_headway,
H_U^{prev},
\Lambda^L_{now},
\nabla\Lambda^L,
\hat{\Lambda}^L_{0:15},
\hat{\Lambda}^L_{15:30},
\hat{\Lambda}^L_{30:60},
\sigma^L,
E^M,
E^H,
HoldFB^L,
DriftFB
]
]

其中：

[
\hat{\Lambda}^L_{0:15}
======================

\int_t^{t+15min}\hat{\lambda}^L(\tau)d\tau
]

[
E^H_t
=====

\sum_{\tau=t-W}^{t}
(\lambda^H_\tau)^2
]

上层可以知道“高频能量很强”，但不要直接看到每个站点的高频 residual 序列。原因是：如果上层直接看 raw high-frequency demand，它会学会用 timetable 对局部 burst 过拟合，最后又回到上下层职责混乱。

上层可以看三类高频摘要：

[
E^H=\text{高频能量}
]

[
persist^H=\text{高频持续时间}
]

[
unserved^H=\text{下层没消化掉的高频积压}
]

这允许高频向低频转化，但不会让上层直接做下层该做的事。

---

## 6. 下层 state：高频为主，但必须知道上层目标

低层原本 state 已经有 bus_id、station_id、hour、direction、forward/backward headway、dwell estimate、headway deviation、recent speeds 等，paper 里明确 lower action 是 holding time，reward/cost 主要看 headway deviation。 新版要把局部高频 demand residual 加进去。

新的下层 state：

[
s^L_t=
[
s^L_{old},
H_U(t,dir),
\Delta h,
load,
capacity_remaining,
q_{station},
\lambda^H_{station,t},
E^H_{station,t},
speed^H_{segment,t},
shock_age,
schedule_slack
]
]

其中 `shock_age` 很重要：

[
shock_age_t =
#{\tau<t: |\lambda^H_\tau|>\kappa}
]

它告诉低层：这是一个瞬时 burst，还是正在变成 regime shift 的持续冲击。

低层可以看到上层目标 (H_U)，但不能看到完整低频 forecast。它的任务是：

[
\pi_L:
(s^L_t, H_U(t), \lambda^H_t)
\mapsto
a^L_t
]

也就是在上层 timetable 给定的服务结构内做动态 correction。

---

## 7. 关键机制：高频可以升格为低频，但要通过 gate，不是直接乱流

你说“不一定要上下层完全独立”，“有点 non-stationary 那意思”，这个判断很对。所以要做一个 **Cross-frequency Promotion Gate**。

定义高频残差的持续能量：

[
P^H_t
=====

\sum_{\tau=t-W}^{t}
\mathbb{1}(|r_\tau|>\kappa)
]

如果：

[
P^H_t > \rho W
]

说明高频冲击不是偶然 burst，而是 regime shift。此时做三件事：

第一，给上层一个 `promotion_flag`：

[
g^{promote}_t=1
]

第二，提高低频状态模型的 process noise：

[
Q^L_t \leftarrow \alpha_Q Q^L_t
]

这样 low-frequency estimator 会更快吸收这个变化。

第三，提前触发上层 replanning：

[
t_U^{next} \leftarrow t
]

这就是“高频向低频转化”。

反过来，如果下层长期 holding drift 很大，也应该反馈给上层：

[
D^{hold}_t
==========

\text{LPF}
\left(
\sum_{\tau=t-W}^{t}a^L_\tau
\right)
]

如果 (D^{hold}_t) 持续偏正，说明下层一直在用 holding 补上层计划不足；这不是下层应该长期承担的工作，应该进入上层 state 作为 `DriftFB`，让上层调整 timetable。

你的现有 TransitDuet 已经有 HoldFB，且 ablation 显示 HoldFB 很重要；paper 里写到 HoldFB 让上层知道低层是否在过度干预，避免系统性 over-holding。 新版可以把 HoldFB 升级成 **frequency-aware HoldFB**：

[
HoldFB =
[
\bar{a}^L,
std(a^L),
LPF(cumsum(a^L)),
HPF(a^L),
unserved^H
]
]

---

## 8. 防止下层“篡改时刻表”：必须加 lower low-frequency leakage penalty

这是你这个问题里最关键、也最有论文价值的地方。

你说得对：在第二个站 holding 1 minute，效果上就是把后续所有站点时刻表向后延迟 1 minute。也就是说，低层 holding 的累积和会形成低频 schedule drift。

所以不能只分解 demand，还要分解 action effect。

低层 action 的累计影响：

[
A^L_{cum}(t)
============

\sum_{\tau \le t}a^L_\tau
]

如果 (A^L_{cum}) 的低频成分太大，说明下层正在做上层的工作。加入 penalty：

[
\mathcal{L}_{leak}^{L}
======================

\left|
\text{LPF}
\left(
A^L_{cum}
\right)
\right|^2
]

工程上可以先不用真正 LPF，直接用 rolling window zero-drift：

[
\mathcal{L}_{drift}^{L}
=======================

\left(
\frac{1}{W}
\sum_{\tau=t-W}^{t}
a^L_\tau
\right)^2
]

下层仍然可以临时 hold，但不能连续几十分钟都在 hold，把时刻表整体拖慢。

同时上层也要防止高频乱动：

[
\mathcal{L}_{leak}^{U}
======================

|\text{HPF}(H_U)|^2
]

最终的职责边界是：

[
\text{Upper: low-pass action}
]

[
\text{Lower: high-pass correction}
]

这比“上层看低频、下层看高频”更强，因为它约束了 **动作的频率归属**。

---

## 9. 奖励函数：从 headway stability 转向 passenger waiting time，并做频率归因

建议目标函数从现在的 composite 继续扩展。当前 paper 里 composite 已经包含 wait、fleet overshoot、CV，repo/paper 的实验也是围绕 wait、CV、overshoot、composite。 下一版可以把 passenger waiting time 作为主目标，再按频率归因。

### 上层 reward

上层主要优化低频需求下的计划质量：

[
R_U
===

*

## \frac{W^L}{10}

\alpha_f
\frac{overshoot^2}{N_{fleet}}
-----------------------------

## \alpha_{cv}CV

## \alpha_s|\Delta^2H_U|^2

## \alpha_{hf}|\text{HPF}(H_U)|^2

\alpha_{drift}D^{hold}
]

其中：

[
W^L
===

\text{expected waiting time under }\lambda^L
]

近似可以用：

[
W^L
\approx
\sum_{d,t}
\Lambda^L_{d,t}
\cdot
\frac{H_U(t,d)}{2}
]

如果考虑 headway variance：

[
E[wait]
\approx
\frac{E[H^2]}{2E[H]}
]

这会自然惩罚不稳定 headway。

### 下层 reward

下层优化高频 residual 下的边际等待时间改善：

[
R_L
===

*

## \Delta W^H_{local}

\beta_h
\left|
H^{real} - H_U
\right|
-------

## \beta_a a^L

\beta_d
\left(
\frac{1}{W}\sum_{\tau=t-W}^{t}a^L_\tau
\right)^2
]

其中：

[
W^H_{local}
===========

\sum_{station}
\max(0, q_{station,t}-\hat{q}_{station,t}^{L})
\cdot wait
]

也就是只把“超出低频预期的局部积压”算作低层主要责任。

这能避免一个问题：如果低频需求本身就很高，不能让下层通过疯狂 holding 去解决，因为那是上层 timetable/headway plan 的问题。

---

## 10. 训练流程：四阶段，不要一上来端到端乱训

### Phase 0：加日志，不改策略

先在现有 TransitDuet 上记录以下序列：

[
y_{station,dir,t}
]

[
q_{station,t}
]

[
board_{station,t}
]

[
a^L_t
]

[
H^{real}_t
]

[
\delta_t / H_U(t)
]

[
wait_t
]

这一步只改 logging，不改 policy。目的是验证需求频谱、holding 频谱、wait 频谱之间是否真的有对应关系。

### Phase 1：离线训练 demand decomposer

用已有 passenger_OD.xlsx / simulation log 训练：

[
\hat{\lambda}^L,\lambda^H,E^H,promote_flag
]

先不进 RL，只做三个验证图：

第一，(\lambda^L) 是否能跟上早晚高峰。第二，(\lambda^H) 是否能捕捉局部 burst。第三，(\lambda^H) 的能量是否和下层 holding / wait spike 相关。

### Phase 2：Freq-state but old action

先不改上层 action，只把 state 换掉：

[
s^U:
demand
\rightarrow
[\lambda^L,\nabla\lambda^L,\hat{\lambda}^L_{0:15},\hat{\lambda}^L_{15:30},E^H]
]

[
s^L:
s^L_{old}
\rightarrow
[s^L_{old},\lambda^H_{local},E^H_{local},shock_age]
]

这一步最容易验证“频率分离 state 有没有用”。

### Phase 3：上层改成 timetable generator

把上层 action 从 1D (\delta_t) 改成 spline coefficients：

[
a_U \in \mathbb{R}^{6}
]

每个方向 3 个 coefficient。输出未来 45 分钟 headway curve。

MVP 中仍然可以映射回每个 dispatch 的 target headway：

[
target_headway_i = H_U(t_i,dir_i)
]

此时不一定真实改 launch time，但论文可以称为 **target-headway timetable**。

### Phase 4：真实 terminal timetable / 首站 holding

最后再改环境：

[
launch_i^{actual}=\max(ready_i,T_i^{sched})
]

也就是上层真实控制 terminal departure。低层只控制中间站 holding。

这个阶段最贴合你最开始的理论观点：上层时刻表控制就是首站 holding，下层 dynamic control 是中间站 holding。

---

## 11. 代码改造清单

你 repo 现在有 `transit_duet/env`、`transit_duet/lower`、`transit_duet/upper`、`transit_duet/coupling`、`transit_duet/scripts` 等目录。([GitHub][3]) 建议新增和修改如下。

### 新增目录

```text
transit_duet/frequency/
  demand_logger.py
  intensity_estimator.py
  causal_decomposer.py
  promotion_gate.py
  freq_features.py
  diagnostics.py
```

### 新增配置

```yaml
frequency:
  enable: true
  bin_sec: 60
  low_cut_min: 30
  high_cut_min: 5
  model: "dynamic_harmonic_nb"
  fourier_K: 4
  station_level: true
  od_low_rank: true
  promotion:
    enable: true
    window_min: 15
    residual_threshold: 2.0
    persistence_ratio: 0.4
  leakage:
    upper_hf_penalty: 0.01
    lower_lf_drift_penalty: 0.05
```

### upper config 修改

```yaml
upper:
  mode: "timetable_spline"
  state_dim: 32
  action_dim: 6
  horizon_min: 45
  replan_min: 15
  spline_knots: 3
  headway_min: 180
  headway_max: 900
  smooth_penalty: 0.01
```

### lower config 修改

```yaml
lower:
  use_freq_state: true
  add_high_freq_demand: true
  add_local_burst_energy: true
  add_speed_residual: true
  drift_window_min: 20
  drift_penalty: 0.05
```

### runner_v3.py 修改点

你现在的 `runner_v3.py` 已经在 `_upper_callback` 里区分 HIRO mode：(\delta_t) 改 `target_headway`，不改 launch time。([GitHub][1]) 新版可以把这里的：

```python
delta_t = upper_action[0]
trip.target_headway = base_hw + delta_t
```

改成：

```python
coeffs = upper_action
planned_hw = timetable_planner.evaluate(
    coeffs=coeffs,
    time=trip.launch_time,
    direction=trip.direction,
)
trip.target_headway = planned_hw
```

完整版再把：

```python
trip._delta_t = 0
```

改成 terminal dispatch schedule：

```python
trip.scheduled_launch = timetable_planner.next_launch(direction)
trip.launch_time = max(trip.ready_time, trip.scheduled_launch)
```

但这个建议放在第二阶段，避免一开始环境变化太大。

---

## 12. 实验设计：必须证明“频率分工”不是普通多加几个特征

你要设计的 baseline 不是越多越好，而是要能打穿 reviewer 的质疑。

### 必做 baseline

| 名称                | 上层输入                              | 下层输入                  | 目的             |
| ----------------- | --------------------------------- | --------------------- | -------------- |
| TransitDuet       | 原始 demand                         | 原始 local state        | 当前主 baseline   |
| RawHistory        | 过去 (W) 分钟 raw demand              | raw local state       | 证明不是“历史窗口更长”   |
| AllFreq-AllLayers | 上下层都看 (\lambda^L,\lambda^H)       | 上下层都看                 | 证明不是“频域特征都给就好” |
| LF-Upper Only     | 上层看 LF，下层不看 HF                    | 下层原始                  | 测上层低频贡献        |
| HF-Lower Only     | 上层原始，下层看 HF                       | 下层 HF                 | 测下层高频贡献        |
| Swapped           | 上层看 HF，下层看 LF                     | 反过来                   | 证明分工方向正确       |
| FreqTransitDuet   | 上层 LF，下层 HF，带 promotion + leakage | 完整方法                  | 主模型            |
| No-Promotion      | 不允许 HF 升格                         | 测 non-stationary gate |                |
| No-Leakage        | 不惩罚低层 drift                       | 测下层是否篡改 timetable     |                |

### 指标

原有指标继续保留：

[
Wait,\quad CV,\quad Overshoot,\quad Composite
]

新加频率诊断指标：

[
UpperHFPower
============

\frac{|\text{HPF}(H_U)|^2}{|H_U|^2}
]

越低越好，说明上层没有做高频控制。

[
LowerLFDrift
============

\frac{
|\text{LPF}(A^L_{cum})|^2
}{
|A^L_{cum}|^2
}
]

越低越好，说明下层没有长期改时刻表。

[
FocusScore
==========

I(a_U;\lambda^L)-I(a_U;\lambda^H)
+
I(a_L;\lambda^H)-I(a_L;\lambda^L)
]

越高越好，说明上层主要响应低频，下层主要响应高频。

还有一个很重要的 operational 指标：

[
ShockResponseTime
=================

t(\text{holding response})-t(\text{HF shock})
]

这个可以证明下层确实更快响应高频变化。

---

## 13. 论文贡献点可以这样写

这条线的贡献不是“用了小波”或“用了 Fourier”，而是：

**Contribution 1：Frequency-separated hierarchical control.**
提出一种面向外生时间序列环境的 HRL 分工原则：低频状态驱动 high-level planning，高频 residual 驱动 low-level dynamic control。

**Contribution 2：Causal demand decomposition for online RL.**
提出 causal demand intensity decomposition，避免未来信息泄漏，并把 cutoff 与上下层决策周期绑定。

**Contribution 3：Timetable-as-low-frequency policy.**
把上层从 per-dispatch scalar shift 扩展为 smooth timetable/headway curve generator，使上层 action 自然低通。

**Contribution 4：Holding leakage regularization.**
指出下层 holding 的累计效应会变成低频 timetable drift，并设计 lower low-frequency leakage penalty 防止下层长期篡改上层计划。

**Contribution 5：Cross-frequency promotion for non-stationarity.**
允许持续高频 residual 被升格为低频 regime shift，使系统不是机械切分，而是可以适应非平稳需求。

这几个点比“用了 Haar/EMA”强很多。

---

## 14. 最小可行版本：两周内能跑的 MVP

最推荐你先做这个版本：

第一，bin demand 为 60 秒粒度，按 station-direction 聚合。

第二，用 causal dynamic harmonic smoother 得到：

[
\lambda^L,\quad r^H,\quad E^H
]

如果实现 state-space 来不及，第一版用 causal spline + rolling residual 也可以，但接口保持不变。

第三，上层 state 替换为：

[
[
old_upper_state\setminus demand,
\lambda^L_{now},
\nabla\lambda^L,
\hat{\Lambda}^L_{0:15},
\hat{\Lambda}^L_{15:30},
E^H
]
]

第四，下层 state 加：

[
[
r^H_{station},
E^H_{station},
shock_age
]
]

第五，先保留 HIRO coupling，只把 (\delta_t) 解释为低频 target-headway adjustment，并加：

[
\mathcal{L}*{U,HF}
+
\mathcal{L}*{L,drift}
]

第六，对比：

[
TransitDuet,\ RawHistory,\ AllFreq,\ Swapped,\ FreqTransitDuet
]

如果这一版能赢，第二版再把上层 action 改成 spline timetable generator。第三版再把 launch time 改成真实 terminal dispatch。

---

## 15. 我建议的最终方法一句话

**上层不是简单看低频特征，而是输出一个低通、平滑、可执行的 timetable/headway curve；下层不是简单看高频特征，而是在这个 timetable 下对局部高频 residual 做快速 holding correction；持续高频冲击通过 promotion gate 进入上层，低层长期 holding drift 通过 leakage feedback 反向提醒上层重规划。**

这就是你想要的“上层专注低频深层规律，下层专注高频快速变化，但允许非平稳转化”的完整施工路线。

[1]: https://github.com/erzhu419/TransitDuet/blob/master/transit_duet/runner_v3.py "TransitDuet/transit_duet/runner_v3.py at master · erzhu419/TransitDuet · GitHub"
[2]: https://github.com/erzhu419/TransitDuet/blob/master/transit_duet/config_v2.yaml "TransitDuet/transit_duet/config_v2.yaml at master · erzhu419/TransitDuet · GitHub"
[3]: https://github.com/erzhu419/TransitDuet/tree/master/transit_duet "TransitDuet/transit_duet at master · erzhu419/TransitDuet · GitHub"
