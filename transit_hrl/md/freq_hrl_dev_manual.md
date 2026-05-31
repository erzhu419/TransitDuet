# Freq-HRL / FreqDuet-General 开发手册

**版本**：v0.1  
**目标**：把当前 TransitDuet 中“分频有效”的工程结果升级为一个通用的、可复现的、理论上有新意的 **Frequency-Separated Hierarchical Reinforcement Learning** 框架。  
**核心原则**：最小化歧义性，最大化创新型理论性。任何实现必须优先满足本手册中的接口契约、因果性约束、频率职责边界、promotion 机制、action-effect leakage 约束和诊断指标。

---

## 0. 一句话定义

**Freq-HRL 是一种面向非平稳外生时间序列控制问题的层级强化学习框架：它把外生信号在线因果分解为低频趋势、中频 regime buffer 和高频 residual，并把低频趋势绑定到高层 planning，把高频 residual 绑定到低层 dynamic control，同时用 cross-frequency promotion 允许短期冲击升格为长期 regime，用 action-effect leakage regularization 防止上下层越权。**

该方法不是：

```text
Wavelet/PINN feature engineering + existing HRL
```

而是：

```text
frequency decomposition as hierarchical control responsibility
```

也就是说，分解器只是可替换模块，真正的算法贡献是：

1. 频率分量如何分配给 HRL 上下层；
2. 高频如何通过 gate 升格为低频 regime；
3. 下层动作如何被防止累积成高层计划漂移；
4. 频率诊断如何证明该方法不是“多加几个特征”。

---

## 1. 命名和范围

### 1.1 通用算法名

正式算法名：

```text
Freq-HRL: Frequency-Separated Hierarchical Reinforcement Learning
```

完整论文名可写为：

```text
Frequency-Separated Hierarchical Reinforcement Learning for Non-stationary Exogenous Time-Series Control
```

### 1.2 领域实例名

公交实例：

```text
FreqTransitDuet
```

量化交易实例：

```text
FreqTradeDuet
```

通用代码包：

```text
freq_hrl/
```

当前 TransitDuet 仓库内的工程落点：

```text
FreqDuet/freqduet/
```

### 1.3 本手册解决的问题

本手册解决的问题是：

```text
在随机、非平稳、噪声强、输入为外生时间序列的 RL / HRL 环境中，
如何把外生状态的频率结构变成上下层策略分工、信用分配、动作约束和非平稳适应机制。
```

### 1.4 本手册不解决的问题

本手册不把以下内容作为主贡献：

1. 不是为了证明 Haar 比 EMA 好；
2. 不是为了证明 PINN 比 state-space model 好；
3. 不是为了做一个更复杂的 demand predictor 或 price predictor；
4. 不是为了把所有频域特征全部喂给一个策略网络；
5. 不是为了让上层和下层完全独立。

PINN、小波、傅里叶、state-space、EMA、learnable wavelet 都只是 `CausalSpectralEncoder` 的候选实现。主贡献必须保持在 **Freq-HRL 控制协议** 上。

---

## 2. 形式化问题定义

### 2.1 系统变量

一个 Freq-HRL 环境由两类状态组成：

```text
z_t: endogenous state, 系统内部状态
x_t: exogenous stream, 外生时间序列输入
```

其中：

```text
z_t = environment state affected by actions
x_t = external time-series driver not directly controlled by the agent
```

Transit 示例：

```text
z_t = bus positions, loads, queues, headways, fleet usage
x_t = passenger demand, OD flow, station arrival counts, route speed shock
```

Quant 示例：

```text
z_t = position, cash, inventory, risk exposure, outstanding orders
x_t = returns, volume, spread, depth, imbalance, volatility, news signal
```

### 2.2 异步层级决策

高层和低层不要求共享 fixed clock。

高层决策时间集合：

```text
T_U = {t_0^U, t_1^U, ..., t_K^U}
```

低层决策时间集合：

```text
T_L = {t_0^L, t_1^L, ..., t_N^L}
```

通常满足：

```text
|T_L| >> |T_U|
```

并且一个高层 action 生效期间允许多个低层 action。

### 2.3 因果外生信号分解

Freq-HRL 必须使用 causal decomposition：

```math
E_\phi(x_{\le t}) \rightarrow
\left(
    x^L_t,
    x^M_t,
    x^H_t,
    \hat{x}^L_{t:t+H},
    \Sigma^L_t,
    E^H_t,
    p^H_t
\right)
```

其中：

```text
x^L_t: low-frequency trend / plan-level signal
x^M_t: mid-frequency regime buffer
x^H_t: high-frequency innovation / residual
hat{x}^L_{t:t+H}: low-frequency causal forecast
Sigma^L_t: low-frequency uncertainty
E^H_t: high-frequency energy summary
p^H_t: high-frequency persistence summary
```

硬约束：

```math
E_\phi(x_{\le t}) \text{ must not use } x_{t+1:T}
```

任何使用完整 episode、完整交易日、完整服务日进行离线分解并把结果回填给当前策略的做法，都必须标记为 **non-causal diagnostic only**，不能作为主实验方法。

### 2.4 频率分工式策略

高层策略：

```math
a^U_k = \pi_U\left(
    z^U_{t_k},
    x^L_{t_k},
    \hat{x}^L_{t_k:t_k+H},
    \Sigma^L_{t_k},
    E^H_{t_k},
    p^H_{t_k},
    g^{promote}_{t_k},
    \ell^{feedback}_{t_k}
\right)
```

低层策略：

```math
a^L_t = \pi_L\left(
    z^L_t,
    a^U_{\kappa(t)},
    x^H_t,
    x^M_t,
    e_t,
    shock\_age_t
\right)
```

其中：

```text
kappa(t): 当前低层时刻 t 所属的高层 plan index
e_t: 当前系统执行误差，如 headway deviation 或 inventory-target gap
```

上层只允许看到高频摘要，不允许看到 raw high-frequency sequence。低层必须知道当前高层目标，但不允许看到完整低频 forecast。

---

## 3. 不可违反的设计原则

### Rule 1: Causality first

任何输入到 policy 的频率分量都必须由当前及过去数据计算。

禁止：

```text
使用整天需求曲线做 DWT/STL/Fourier，然后把当前时刻的 LF/HF 结果喂给 policy。
```

允许：

```text
trailing window EMA / causal wavelet / online state-space / recursive Fourier / online Kalman / online particle filter
```

### Rule 2: cutoff 由控制周期定义，而不是由滤波器习惯定义

频段不是先验固定的数学概念，而是相对于控制任务定义的职责边界。

```text
低频 = 高层计划应该吸收的变化
高频 = 低层动态控制应该响应的变化
中频 = 可能正在从 shock 转化为 regime 的缓冲区
```

### Rule 3: upper receives LF + compressed HF summaries only

上层输入必须满足：

```text
must contain: low trend, low forecast, low uncertainty, high-energy summary, high-persistence summary, promotion flag, leakage feedback
must not contain: raw high-frequency residual sequence at station/tick granularity
```

原因：如果上层能直接看 raw HF，它会学会用低频 plan 去追高频噪声，导致上层高频抖动。

### Rule 4: lower receives HF + current plan only

低层输入必须满足：

```text
must contain: current high-level target/plan, local HF residual, local mid-frequency regime context, execution error
must not contain: full low-frequency forecast horizon
```

原因：如果低层能看到完整 LF forecast，它会长期替上层修 plan，导致 low-frequency action leakage。

### Rule 5: cross-frequency conversion must pass through gate

高频不能直接进入上层决策。高频升格必须经过：

```text
High residual -> persistence test -> promotion gate -> upper replan / low-state update
```

### Rule 6: decompose action effects, not only inputs

只分解外生输入不够。必须定义动作影响算子：

```math
\mathcal{G}_U(a^U), \quad \mathcal{G}_L(a^L)
```

并约束：

```text
upper action effect should be low-pass
lower cumulative action effect should not contain persistent low-frequency drift
```

### Rule 7: decomposer is replaceable; routing protocol is not

可以替换 decomposer：

```text
EMA, causal moving average, STFT, causal wavelet, MODWT, dynamic Fourier, state-space, PINN, learnable wavelet
```

不可以替换核心控制协议：

```text
LF -> upper planner
HF -> lower controller
persistent HF -> promotion gate
action-effect leakage -> regularizer + feedback
```

### Rule 8: diagnostics are mandatory

没有频率诊断，不能声称方法有效。任何主实验必须同时报告：

```text
Task metrics + Frequency responsibility metrics + Leakage metrics + Promotion metrics
```

---

## 4. 总体模块架构

### 4.1 模块图

```text
Raw Environment Stream
        |
        v
ExogenousStreamAdapter
        |
        v
CausalSpectralEncoder
        |
        +--------------------+
        |                    |
        v                    v
FrequencyRouter       PromotionGate
        |                    |
        +---------+----------+
                  |
          +-------+-------+
          |               |
          v               v
HighLevelPlanner   LowLevelController
          |               |
          v               v
   Plan / Target     Residual Correction
          |               |
          +-------+-------+
                  |
                  v
        Domain Environment
                  |
                  v
ActionEffectOperator
                  |
                  v
LeakageRegularizer + FrequencyDiagnostics
```

### 4.2 必须实现的核心接口

#### ExogenousStreamAdapter

职责：把领域原始数据转换成统一外生流。

```python
class ExogenousStreamAdapter:
    def reset(self, episode_id: int) -> None:
        ...

    def observe(self, raw_event: dict, t: float) -> None:
        ...

    def get_bin(self, t: float) -> dict:
        """
        Return causal aggregated features for the current time bin.
        Must only use events with timestamp <= t.
        """
        ...

    def get_schema(self) -> dict:
        ...
```

输出 schema 必须包含：

```text
timestamp
entity_id
x_raw
valid_mask
normalization_context
```

Transit entity examples：

```text
route_direction, station_direction, OD_factor
```

Quant entity examples：

```text
symbol, asset_group, market_regime_bucket
```

#### CausalSpectralEncoder

职责：在线因果分解外生流。

```python
class CausalSpectralEncoder:
    def reset(self, episode_id: int) -> None:
        ...

    def update(self, x_bin: dict, t: float) -> None:
        ...

    def features(self, t: float) -> dict:
        """
        Return only causal frequency features.
        """
        return {
            "x_low": ...,             # low-frequency current estimate
            "x_low_slope": ...,
            "x_low_forecast": ...,    # forecast bins
            "x_low_uncertainty": ...,
            "x_mid": ...,
            "x_high": ...,            # local residual
            "x_high_energy": ...,
            "x_high_persistence": ...,
            "shock_age": ...,
        }
```

#### FrequencyRouter

职责：把 encoder 输出路由给上层和下层，同时执行信息屏蔽。

```python
class FrequencyRouter:
    def upper_view(self, freq_features: dict, z_upper: dict) -> dict:
        """
        Must include LF and compressed HF summaries.
        Must exclude raw HF residual sequence.
        """
        ...

    def lower_view(self, freq_features: dict, z_lower: dict, current_plan: dict) -> dict:
        """
        Must include local HF and current plan.
        Must exclude full LF forecast horizon.
        """
        ...
```

Router 必须有 mask test：

```text
assert "x_high_sequence" not in upper_state
assert "x_low_forecast_full" not in lower_state
```

#### HighLevelPlanner

职责：输出低通计划，而不是快速反应动作。

```python
class HighLevelPlanner:
    def act(self, s_upper: dict, deterministic: bool = False) -> dict:
        return {
            "plan_params": ...,
            "plan_curve": ...,
            "risk_or_capacity_budget": ...,
            "valid_until": ...,
        }
```

Transit：输出 headway/timetable curve。  
Quant：输出 target portfolio / target inventory / risk budget curve。

#### LowLevelController

职责：在当前高层计划下响应高频 residual。

```python
class LowLevelController:
    def act(self, s_lower: dict, deterministic: bool = False) -> dict:
        return {
            "correction_action": ...,
            "execution_metadata": ...,
        }
```

Transit：输出 station holding。  
Quant：输出 order size / execution speed / quote skew。

#### PromotionGate

职责：判断高频 residual 是否正在变成低频 regime。

```python
class PromotionGate:
    def update(self, freq_features: dict, t: float) -> dict:
        return {
            "promote": bool,
            "promotion_strength": float,
            "reason": str,
            "cooldown_remaining": float,
        }
```

#### ActionEffectOperator

职责：把动作转换成对系统计划的实际影响。

```python
class ActionEffectOperator:
    def upper_effect(self, upper_action_history: list) -> np.ndarray:
        ...

    def lower_effect(self, lower_action_history: list) -> np.ndarray:
        ...
```

Transit：

```text
lower_effect = cumulative holding-induced downstream schedule drift
upper_effect = planned headway/timetable curve variation
```

Quant：

```text
lower_effect = inventory / position drift caused by executions
upper_effect = target weight / risk budget variation
```

#### LeakageRegularizer

职责：约束动作影响的频率归属。

```python
class LeakageRegularizer:
    def compute(self, upper_effect, lower_effect) -> dict:
        return {
            "upper_hf_penalty": ...,
            "lower_lf_penalty": ...,
            "leakage_feedback": ...,
        }
```

#### FrequencyDiagnostics

职责：记录理论指标，证明上下层确实专注于不同频率。

```python
class FrequencyDiagnostics:
    def log_step(self, t, states, actions, freq_features, effects):
        ...

    def summarize_episode(self) -> dict:
        return {
            "UpperHFPower": ...,
            "LowerLFDrift": ...,
            "FocusScore": ...,
            "PromotionDelay": ...,
            "ShockResponseTime": ...,
        }
```

---

## 5. 频率分解器设计

### 5.1 推荐实现顺序

不要一开始上 PINN 或 learnable wavelet。实现顺序必须是：

```text
Level 0: causal rolling / EMA baseline
Level 1: causal dynamic Fourier / harmonic smoother
Level 2: causal state-space model with uncertainty
Level 3: causal wavelet / MODWT trailing-window variant
Level 4: learnable lifting wavelet or neural state-space
Level 5: PINN-constrained encoder, only if physical constraints are explicit
```

### 5.2 默认通用分解

通用外生信号：

```math
x_t = x^L_t + x^M_t + x^H_t
```

推荐初版：

```math
x^L_t = \operatorname{Smooth}_{causal}(x_{\le t}; T_L)
```

```math
r_t = x_t - x^L_t
```

```math
x^H_t = \operatorname{HPF}_{causal}(r_{\le t}; T_H)
```

```math
x^M_t = x_t - x^L_t - x^H_t
```

其中：

```text
T_L: low-frequency cutoff period
T_H: high-frequency cutoff period
```

### 5.3 cutoff 选择规则

定义：

```text
Delta_U = median interval between high-level decisions
Delta_L = median interval between low-level decisions
H_U = high-level planning horizon
```

推荐：

```math
T_H = \min(5\Delta_L, 0.25\Delta_U)
```

```math
T_L = \max(4\Delta_U, 0.5H_U)
```

```math
T_M \in (T_H, T_L)
```

当领域有明确业务尺度时，允许覆盖默认值，但必须记录理由。

Transit 初始值：

```yaml
bin_sec: 60
high_cut_min: 5
low_cut_min: 30
mid_band_min: [5, 30]
```

Quant 初始值：

```yaml
bar_sec: 60
lower_step_bar: 1
upper_rebalance_min: 60
high_cut_min: 5
low_cut_min: 120
mid_band_min: [5, 120]
```

### 5.4 因果性测试

必须实现以下测试：

```python
def test_encoder_is_causal():
    y1 = encoder.run(x[:t])
    y2 = encoder.run(concat(x[:t], random_future_noise))
    assert_allclose(y1.features_at(t), y2.features_at(t))
```

任何不能通过该测试的 encoder 不允许进入主实验。

---

## 6. Policy state / action 契约

### 6.1 高层 state 契约

高层 state 必须是：

```math
s^U_t = [
    z^U_t,
    x^L_t,
    \nabla x^L_t,
    \hat{x}^L_{t:t+H},
    \Sigma^L_t,
    E^M_t,
    E^H_t,
    p^H_t,
    g^{promote}_t,
    \ell^{feedback}_t
]
```

高层 state 禁止包含：

```text
raw high-frequency residual sequence
station-level / tick-level HF sequence unless aggregated into bounded summaries
future decomposed value computed using x_{>t}
```

### 6.2 低层 state 契约

低层 state 必须是：

```math
s^L_t = [
    z^L_t,
    a^U_{\kappa(t)},
    target\_error_t,
    x^H_{local,t},
    \Delta x^H_{local,t},
    E^H_{local,t},
    x^M_{local,t},
    shock\_age_t
]
```

低层 state 禁止包含：

```text
full low-frequency forecast horizon
high-level value estimate
future low-frequency trend
```

### 6.3 高层 action 契约

高层 action 必须是计划，不是单点修正。

通用形式：

```math
a^U_k = \theta_k
```

其中 `theta_k` 参数化一个未来计划曲线：

```math
P_U(\tau) = P_0(\tau) + \sum_{m=1}^{M} \theta_{k,m} B_m(\tau), \quad \tau \in [t_k, t_k + H_U]
```

Transit：

```text
P_U(t) = planned headway / timetable curve
```

Quant：

```text
P_U(t) = target weight / target inventory / risk budget curve
```

### 6.4 低层 action 契约

低层 action 必须是 residual correction。

Transit：

```math
a^L_t = holding\_time_t
```

Quant：

```math
a^L_t = [order\_size_t, execution\_speed_t, limit/market\_choice_t, quote\_skew_t]
```

低层 action 允许短期偏离高层计划，但不允许长期改变高层计划的低频含义。

---

## 7. Cross-frequency Promotion Gate

### 7.1 目标

Promotion gate 解决的问题是：

```text
一个高频 shock 如果持续存在，就不再是 noise，而是新的 low-frequency regime。
```

Transit 示例：某站点连续出现异常客流，变成新的需求模式。  
Quant 示例：新闻冲击导致 order-flow / volatility 持续异常，变成 trend 或 volatility regime。

### 7.2 触发条件

定义标准化高频 residual：

```math
r^H_t = \frac{x_t - \hat{x}^L_t}{\sqrt{\operatorname{Var}(x_t \mid x^L_t) + \epsilon}}
```

定义持续性：

```math
P^H_t = \sum_{\tau=t-W}^{t} \mathbf{1}\left(|r^H_\tau| > \kappa_r \lor E^H_\tau > \kappa_E\right)
```

触发条件：

```math
P^H_t > \rho W
```

默认参数：

```yaml
promotion:
  window: 15min
  residual_threshold: 2.0
  energy_threshold_quantile: 0.90
  persistence_ratio: 0.40
  cooldown: 30min
```

### 7.3 触发后的强制行为

Promotion 一旦触发，必须执行三件事：

1. 给高层 state 注入：

```text
g_promote = 1
promotion_strength > 0
promotion_reason
```

2. 提高低频模型更新速度：

```math
Q^L_t \leftarrow \alpha_Q Q^L_t, \quad \alpha_Q > 1
```

3. 触发高层提前 replan：

```math
t^U_{next} \leftarrow t
```

Quant 额外动作：

```math
risk\_budget_t \leftarrow \gamma_{risk} risk\_budget_t, \quad 0 < \gamma_{risk} \le 1
```

Transit 额外动作：

```text
allow high-level headway/timetable curve to update before scheduled replan time
```

### 7.4 防抖规则

Promotion gate 必须有 hysteresis 和 cooldown。

```yaml
promotion:
  enter_ratio: 0.40
  exit_ratio: 0.20
  min_duration: 3 bins
  cooldown: 30min
```

没有 cooldown 的 promotion gate 会把高层变成高频控制器。

---

## 8. Action-effect leakage regularization

### 8.1 为什么必须约束 action effect

Freq-HRL 的创新点之一是：不仅分解输入，还要分解动作影响。

因为低层动作虽然是高频的，但它的累计影响可能是低频的。

Transit：

```text
连续 station holding -> downstream schedule drift
```

Quant：

```text
连续 execution buy/sell -> inventory / position drift
```

### 8.2 通用定义

定义动作影响：

```math
e^U_t = \mathcal{G}_U(a^U_{0:k})
```

```math
e^L_t = \mathcal{G}_L(a^L_{0:t})
```

上层 leakage：

```math
\mathcal{L}_{U,HF} = \left\| HPF(e^U_{0:t}) \right\|_2^2
```

低层 leakage：

```math
\mathcal{L}_{L,LF} = \left\| LPF(e^L_{0:t}) \right\|_2^2
```

总 leakage：

```math
\mathcal{L}_{leak} = \lambda_U \mathcal{L}_{U,HF} + \lambda_L \mathcal{L}_{L,LF}
```

### 8.3 工程近似

如果暂时不实现 HPF/LPF，可以用 rolling window 近似。

上层平滑约束：

```math
\mathcal{L}_{U,smooth} = \sum_k \left\| \Delta^2 P_U(t_k) \right\|^2
```

低层零漂移约束：

```math
\mathcal{L}_{L,drift} = \left( \frac{1}{W} \sum_{\tau=t-W}^{t} e^L_\tau \right)^2
```

### 8.4 Domain-specific operators

Transit：

```math
e^L_t = \sum_{\tau \le t} holding_\tau
```

```math
\mathcal{L}^{Transit}_{L,drift} = \left\| LPF\left(\sum_{\tau \le t} holding_\tau\right) \right\|^2
```

Quant：

```math
I_t = I_{t-1} + q_t
```

```math
\mathcal{L}^{Trade}_{L,drift} = \left\| LPF(I_t - I^{target}_t) \right\|^2
```

```math
\mathcal{L}^{Trade}_{U,HF} = \left\| HPF(w^{target}_t) \right\|^2
```

---

## 9. Reward and credit attribution

### 9.1 通用 reward

总 reward：

```math
R_t = R^{task}_t - \lambda_{leak}\mathcal{L}_{leak,t} - \lambda_{switch}\mathcal{L}_{switch,t} - \lambda_{risk}\mathcal{L}_{risk,t}
```

但高层和低层的 credit 必须分频。

### 9.2 高层 reward

高层主要对低频供给/配置质量负责。

```math
R^U_k = -C^L_{plan}(x^L, a^U) - \lambda_U \mathcal{L}_{U,HF} - \lambda_{FB}\ell^{feedback}
```

Transit：

```math
C^L_{plan} \approx \sum_{d,t} \Lambda^L_{d,t} \frac{H_U(t,d)}{2}
```

Quant：

```math
C^L_{plan} = -\operatorname{Return}^{L}_{portfolio} + \lambda_{risk}\operatorname{Risk}^{L} + \lambda_{turn}\operatorname{Turnover}^{U}
```

### 9.3 低层 reward

低层主要对高频 residual 的边际修正负责。

```math
R^L_t = -\Delta C^H_{local}(x^H, a^L \mid a^U) - \lambda_L\mathcal{L}_{L,LF} - \lambda_a\|a^L_t\|
```

Transit：

```math
\Delta C^H_{local} = \text{local residual waiting cost after holding decision}
```

Quant：

```math
\Delta C^H_{local} = \text{slippage} + \text{short-term inventory risk} + \text{execution cost}
```

### 9.4 Credit attribution rule

Episode-level return 必须拆成：

```text
low-frequency attributable cost
high-frequency attributable cost
leakage attributable cost
promotion adaptation cost
```

禁止把所有 episode return 直接同时回传给上下层而不做 attribution 诊断。可以继续使用 existing hindsight credit，但必须增加 frequency attribution logging。

---

## 10. Transit instance: FreqTransitDuet

### 10.1 Transit 映射

```text
x_t: station-direction arrivals, OD counts, route-direction demand, speed shock
z_t: bus position, load, station queue, forward/backward headway, fleet usage
x^L_t: peak/off-peak trend, directional demand curve, OD low-rank trend
x^H_t: station burst, local queue shock, speed residual
x^M_t: persistent local deviation, event-start regime buffer
```

高层：

```text
Input: low-frequency demand trend + forecast + HF summaries + leakage feedback
Output: planned headway / target timetable curve
```

低层：

```text
Input: station-local HF residual + current planned headway + headway error + load
Output: station holding time
```

### 10.2 当前 repo 的落地路径

当前仓库根目录已有 `FreqDuet/`，并且 `FreqDuet/freqduet/` 是隔离出来的实验副本。后续所有频率分离实验应优先在这里改，不要直接污染原始 `transit_duet/`。

推荐路径：

```text
FreqDuet/freqduet/
  frequency/
  runner_v3.py
  configs_freqduet/
  scripts/
```

### 10.3 Transit MVP state

上层 state：

```text
s_U = [
  old_upper_state_without_raw_demand,
  lambda_L_now,
  grad_lambda_L,
  Lambda_L_0_15,
  Lambda_L_15_30,
  Lambda_L_30_60,
  sigma_L,
  E_M,
  E_H,
  persist_H,
  promote_flag,
  HoldFB_L,
  DriftFB
]
```

下层 state：

```text
s_L = [
  old_lower_state,
  H_U_current,
  headway_error_to_plan,
  station_queue,
  load,
  capacity_remaining,
  lambda_H_station,
  delta_lambda_H_station,
  E_H_station,
  speed_H_segment,
  shock_age,
  schedule_slack
]
```

### 10.4 Transit upper action

MVP：仍保持 HIRO-style target-headway channel，不真实改 launch time。

```text
upper action -> planned target headway H_U(t)
trip.target_headway = H_U(trip.launch_time, direction)
trip.launch_time unchanged
```

完整版：真实 terminal dispatch retiming。

```math
launch_i^{actual} = \max(ready_i, T_i^{sched})
```

### 10.5 Transit config

```yaml
frequency:
  enable: true
  bin_sec: 60
  low_cut_min: 30
  high_cut_min: 5
  mid_band_min: [5, 30]
  model: "dynamic_harmonic"
  causal_only: true
  station_level: true
  od_low_rank: true

promotion:
  enable: true
  window_min: 15
  residual_threshold: 2.0
  persistence_ratio: 0.4
  cooldown_min: 30
  increase_low_process_noise: 4.0
  trigger_upper_replan: true

leakage:
  enable: true
  upper_hf_penalty: 0.01
  lower_lf_drift_penalty: 0.05
  drift_window_min: 20

upper:
  mode: "timetable_spline_hiro"
  state_dim: 32
  action_dim: 6
  horizon_min: 45
  replan_min: 15
  spline_knots: 3
  headway_min: 180
  headway_max: 900
  smooth_penalty: 0.01

lower:
  use_freq_state: true
  add_high_freq_demand: true
  add_local_burst_energy: true
  add_speed_residual: true
  add_shock_age: true
```

---

## 11. Quant instance: FreqTradeDuet

### 11.1 Quant 映射

```text
x_t: returns, volume, spread, depth, imbalance, realized volatility, news/funding signal
z_t: cash, position, inventory, leverage, drawdown, outstanding orders, risk exposure
x^L_t: trend, cycle, slow volatility, macro/regime, liquidity regime
x^H_t: return innovation, order-flow shock, spread jump, depth shock, news residual
x^M_t: persistent flow imbalance, emerging volatility regime
```

高层：

```text
Input: trend/regime/slow volatility + risk state + HF summaries
Output: target portfolio / target inventory / risk budget / rebalance horizon
```

低层：

```text
Input: order-flow residual + current target + spread/depth + inventory gap
Output: order size / execution speed / limit-market choice / quote skew
```

### 11.2 Quant MVP environment

第一版不要做 tick-level HFT。先做分钟级或小时级 portfolio + execution。

环境状态：

```text
price_bar_t
volume_t
spread_proxy_t
realized_vol_t
position_t
cash_t
target_position_t
```

高层动作：

```math
a^U_k = [w^{target}_k, risk\_budget_k, rebalance\_horizon_k]
```

低层动作：

```math
a^L_t = \alpha_t \left(w^{target}_{\kappa(t)} - w_t\right)
```

其中：

```text
alpha_t in [0, 1] is execution speed
```

reward：

```math
R_t = portfolio\_return_t - transaction\_cost_t - drawdown\_penalty_t - inventory\_drift\_penalty_t
```

### 11.3 Quant config

```yaml
market:
  bar_sec: 60
  assets: ["asset_0", "asset_1", "asset_2"]
  transaction_cost_bps: 5
  slippage_model: "sqrt_volume"
  max_leverage: 1.0

frequency:
  enable: true
  low_cut_min: 120
  high_cut_min: 5
  mid_band_min: [5, 120]
  model: "causal_state_space_vol_trend"
  causal_only: true

upper:
  mode: "portfolio_risk_curve"
  rebalance_min: 60
  horizon_min: 240
  action_dim: "n_assets + risk_budget + urgency"
  turnover_penalty: 0.01
  upper_hf_penalty: 0.05

lower:
  mode: "execution_residual_controller"
  step_min: 1
  execution_speed_min: 0.0
  execution_speed_max: 1.0
  inventory_drift_penalty: 0.05

promotion:
  enable: true
  window_min: 30
  residual_threshold: 2.0
  volatility_threshold_quantile: 0.90
  persistence_ratio: 0.4
  risk_shrink: 0.75
```

### 11.4 Quant leakage

```math
\mathcal{L}^{Trade}_{upper} = \frac{\|HPF(w^{target})\|^2}{\|w^{target}\|^2 + \epsilon}
```

```math
\mathcal{L}^{Trade}_{lower} = \frac{\|LPF(I_t - I^{target}_t)\|^2}{\|I_t - I^{target}_t\|^2 + \epsilon}
```

---

## 12. 训练流程

### Phase 0: Logging-only audit

不改策略，只加日志。

必须记录：

```text
x_raw_t
x_bin_t
z_t
a_U_t
a_L_t
plan_curve_t
action_effect_upper_t
action_effect_lower_t
task_reward_t
waiting / PnL / cost decomposition
```

验收：

```text
can reconstruct one episode's frequency features offline from logged causal bins
no missing timestamps
all logs have domain entity id
```

### Phase 1: Causal encoder MVP

实现：

```text
ExogenousStreamAdapter
CausalSpectralEncoder
FrequencyRouter
```

验收：

```text
test_encoder_is_causal passes
x_low smoother than x_raw
x_high has near-zero rolling mean under stationary period
x_high energy spikes around known demand/market shocks
```

### Phase 2: Frequency state, old actions

不改 action space。只替换 state。

Transit：

```text
upper δ_t remains target-headway shift
lower holding remains unchanged
```

Quant：

```text
upper target position remains simple scalar/vector
lower execution remains alpha execution speed
```

验收：

```text
Freq-state > RawHistory in at least one noisy / non-stationary setting
AllFreq-AllLayers does not dominate Freq-routed version
Swapped worsens frequency diagnostics
```

### Phase 3: Leakage regularization

加入：

```text
upper_hf_penalty
lower_lf_drift_penalty
leakage_feedback to upper
```

验收：

```text
UpperHFPower decreases
LowerLFDrift decreases
Task reward does not collapse
```

### Phase 4: High-level plan curve

将高层 action 从单点修正升级为曲线参数。

Transit：

```text
spline headway / timetable curve
```

Quant：

```text
target weight / target inventory curve
```

验收：

```text
plan curve is smooth
upper action frequency remains low
lower can still respond quickly to shocks
```

### Phase 5: Promotion gate

实现 high residual persistence -> promotion。

验收：

```text
short isolated shocks do not trigger promotion
persistent shocks trigger promotion within bounded delay
promotion reduces post-shift recovery cost
No-Promotion ablation adapts slower
```

### Phase 6: Dual-domain validation

在 Transit 和 Quant 两个环境中共用核心模块，只替换 domain adapter 和 action effect operator。

验收：

```text
same core freq_hrl modules used in both domains
domain-specific code only appears under domains/transit and domains/trading
metrics are domain-specific but frequency diagnostics are homologous
```

### Phase 7: Advanced encoder

在核心协议稳定后再加入：

```text
causal MODWT
learnable wavelet
neural state-space
PINN-constrained encoder
```

验收：

```text
advanced encoder improves over causal EMA/state-space baseline
but method still works when encoder is swapped back to simple baseline
```

---

## 13. 实验矩阵

### 13.1 必做 baseline

| 名称 | 频率输入 | 层级结构 | 目的 |
|---|---|---|---|
| Vanilla RL | raw state | no HRL | 普通 RL baseline |
| HRL-Raw | raw history | upper/lower | 证明 HRL 本身不够 |
| RawHistory | trailing raw sequence | upper/lower | 证明不是历史窗口更长 |
| Freq-SinglePolicy | LF/HF all to one policy | no responsibility split | 证明不是频域特征本身 |
| AllFreq-AllLayers | LF/HF all to both layers | HRL | 证明不是全部给最好 |
| LF-Upper Only | LF to upper, raw lower | HRL | 测低频上层贡献 |
| HF-Lower Only | raw upper, HF to lower | HRL | 测高频下层贡献 |
| Swapped | HF to upper, LF to lower | HRL | 证明分工方向正确 |
| No-Promotion | no HF->LF gate | HRL | 测非平稳适应 |
| No-Leakage | no action-effect constraint | HRL | 测越权控制 |
| Freq-HRL | routed LF/HF + promotion + leakage | HRL | 主方法 |

### 13.2 分解器 ablation

| 名称 | 是否 causal | 角色 |
|---|---:|---|
| Causal EMA | yes | 最强简单 baseline |
| Causal Dynamic Fourier | yes | 推荐默认 |
| Causal State-Space | yes | 推荐主方法 encoder |
| Causal Haar trailing-window | yes | wavelet-style robust baseline |
| Non-causal STL/DWT | no | diagnostic only，不可做主结果 |
| Learnable Wavelet | yes if implemented online | advanced |
| PINN-constrained Encoder | yes if online | advanced / physics-constrained |

### 13.3 环境压力测试

必须至少包含：

```text
stationary low-noise
stationary high-noise
non-stationary trend shift
localized shock burst
persistent shock -> regime shift
out-of-distribution demand / market period
```

---

## 14. 诊断指标

### 14.1 通用指标

上层高频动作比例：

```math
UpperHFPower = \frac{\|HPF(e^U)\|^2}{\|e^U\|^2 + \epsilon}
```

低层低频漂移比例：

```math
LowerLFDrift = \frac{\|LPF(e^L)\|^2}{\|e^L\|^2 + \epsilon}
```

频率专注度：

```math
FocusScore = I(a_U; x^L) - I(a_U; x^H) + I(a_L; x^H) - I(a_L; x^L)
```

Promotion 延迟：

```math
PromotionDelay = t(g^{promote}=1) - t(regime\_shift)
```

Shock 响应时间：

```math
ShockResponseTime = t(a_L \text{ responds}) - t(x^H \text{ shock})
```

### 14.2 Transit task metrics

```text
Average passenger waiting time
Headway CV
Fleet overshoot
Composite score
UpperHFPower
LowerLFDrift
FocusScore
ShockResponseTime
PromotionDelay
```

### 14.3 Quant task metrics

```text
Sharpe
Sortino
MaxDrawdown
Calmar
Turnover
TransactionCost
Slippage
InventoryRisk
UpperHFTradeRatio
LowerLFInventoryDrift
RegimePromotionDelay
FrequencyFocusScore
```

### 14.4 论文中必须呈现的图

至少包含：

1. 原始外生信号、低频分量、高频 residual 对照图；
2. 高层 action / plan curve 的频谱图；
3. 低层 cumulative effect 的频谱图；
4. promotion gate 在 regime shift 前后的触发图；
5. ablation bar chart；
6. FocusScore vs task performance scatter；
7. No-Leakage 与 Freq-HRL 的 drift 对比。

---

## 15. 代码结构

### 15.1 通用代码结构

```text
freq_hrl/
  core/
    types.py
    clocks.py
    frequency_router.py
    promotion_gate.py
    leakage.py
    diagnostics.py
    replay.py
  encoders/
    base.py
    causal_ema.py
    causal_fourier.py
    causal_state_space.py
    causal_wavelet.py
    pinn_encoder.py
  policies/
    high_level.py
    low_level.py
    plan_curve.py
  domains/
    transit/
      adapter.py
      action_effect.py
      reward.py
      config_defaults.yaml
    trading/
      adapter.py
      market_env.py
      action_effect.py
      reward.py
      config_defaults.yaml
  experiments/
    transit/
    trading/
  scripts/
    train.py
    evaluate.py
    eval_frequency_modules.py
    plot_diagnostics.py
  tests/
    test_causality.py
    test_router_masks.py
    test_leakage.py
    test_promotion.py
```

### 15.2 当前 TransitDuet repo 内建议结构

```text
FreqDuet/freqduet/
  frequency/
    __init__.py
    stream_adapter.py
    causal_encoder.py
    router.py
    promotion_gate.py
    leakage.py
    diagnostics.py
  domains/
    transit_adapter.py
    transit_action_effect.py
  configs_freqduet/
    F_freqduet_harmonic_hiro.yaml
    F_freqduet_timetable_hiro.yaml
    F_freqduet_ema_hiro.yaml
    F_freqduet_haar_hiro.yaml
    F_ablation_allfreq.yaml
    F_ablation_swapped.yaml
    F_ablation_no_promotion.yaml
    F_ablation_no_leakage.yaml
  scripts/
    eval_frequency_modules.py
    eval_ablation_matrix.py
    plot_frequency_diagnostics.py
```

---

## 16. 测试要求

### 16.1 Causality test

```python
def test_no_future_leakage():
    features_a = run_encoder(x[:t])
    features_b = run_encoder(x[:t] + random_future)
    assert_close(features_a[t], features_b[t])
```

### 16.2 Router mask test

```python
def test_upper_cannot_see_raw_high_sequence():
    s_u = router.upper_view(freq_features, z_upper)
    assert "x_high_raw_sequence" not in s_u
    assert "x_high_local_station_vector" not in s_u
```

```python
def test_lower_cannot_see_full_low_forecast():
    s_l = router.lower_view(freq_features, z_lower, plan)
    assert "x_low_forecast_full" not in s_l
```

### 16.3 Promotion test

```python
def test_promotion_only_for_persistent_shock():
    isolated = make_isolated_spike()
    persistent = make_persistent_shift()
    assert promotion(isolated).triggered is False
    assert promotion(persistent).triggered is True
```

### 16.4 Leakage test

```python
def test_lower_drift_penalty_increases_with_cumulative_bias():
    random_zero_mean = make_zero_mean_actions()
    biased = make_positive_biased_actions()
    assert L_lower(biased) > L_lower(random_zero_mean)
```

### 16.5 Config isolation test

```python
def test_ablation_configs_change_only_intended_switches():
    diff = compare_yaml(base, ablation)
    assert diff.keys() <= allowed_keys_for_that_ablation
```

---

## 17. 常见失败模式

### Failure 1: AllFreq-AllLayers beats Freq-HRL

含义：频率路由没有贡献，可能只是特征不足或 mask 太强。

处理：

```text
检查 upper 是否需要 HF summaries，而不是 raw HF；
检查 lower 是否需要 current plan error；
检查 leakage penalty 是否过强；
检查 cutoff 是否和控制周期不匹配。
```

### Failure 2: Swapped 不变差

含义：频带没有被真正区分，encoder 可能无效。

处理：

```text
检查 x_low 和 x_high 相关性；
检查 high residual 是否近似零均值；
检查 low trend 是否过度滞后；
增加 synthetic regime-shift 测试。
```

### Failure 3: No-Leakage performance 更好

含义：主任务奖励允许下层越权，短期 performance 可能牺牲结构正确性。

处理：

```text
检查长期泛化；
加入 OOD non-stationary test；
报告 LowerLFDrift；
调小而不是删除 leakage penalty。
```

### Failure 4: Promotion 频繁触发

含义：gate 把噪声当成 regime。

处理：

```text
提高 persistence_ratio；
加入 cooldown；
要求 mid-frequency confirmation；
使用 uncertainty-normalized residual。
```

### Failure 5: 高层 plan 过慢，错过变化

含义：低频模型 process noise 过小或 promotion 不敏感。

处理：

```text
提高 Q_L；
缩短 replan interval；
让 upper 看 E_H / persist_H summaries；
启用 promotion-triggered replan。
```

### Failure 6: 低层变成长期规划器

含义：低层可能看到了 LF forecast，或者 drift penalty 太弱。

处理：

```text
检查 router mask；
增加 lower_lf_drift_penalty；
把 DriftFB 反馈给 upper；
限制 lower action cumulative bias。
```

---

## 18. Paper contribution 写法

### Contribution 1: Frequency-separated exogenous-state HRL

提出一个 HRL 分工原则：外生时间序列的低频趋势进入高层 planning，高频 residual 进入低层 dynamic control。

### Contribution 2: Causal spectral encoder for online control

提出可替换的 causal encoder 接口，用于在线分解外生时间序列，避免未来信息泄漏，并把 cutoff 与上下层控制周期绑定。

### Contribution 3: Cross-frequency promotion for non-stationarity

提出高频 residual 持续时升格为低频 regime 的 gate，使方法不是静态频率切分，而能适应非平稳变化。

### Contribution 4: Action-effect frequency leakage regularization

提出不仅分解输入，也分解动作影响；防止高层计划高频抖动，防止低层快速动作累积成长期计划漂移。

### Contribution 5: Dual-domain validation

用 Transit 和 Quant 两个领域验证同一机制：

```text
Transit: timetable/headway planning + station holding
Quant: portfolio/risk planning + execution/inventory correction
```

这会证明方法不是 TransitDuet 特化技巧，而是适用于外生时间序列控制的一般 HRL 原则。

---

## 19. 最小可行版本

### MVP 目标

在当前 FreqDuet 基础上实现一个可跑、可诊断、可 ablation 的版本。

### MVP 必做

1. `bin_sec=60` 的外生流聚合；
2. causal low/high decomposition；
3. upper state 替换为 LF + forecast + HF summaries；
4. lower state 增加 local HF residual + shock_age；
5. 保留 HIRO target-headway coupling；
6. 加 upper HF penalty；
7. 加 lower LF drift penalty；
8. 实现 UpperHFPower、LowerLFDrift、FocusScore；
9. 跑 TransitDuet、RawHistory、AllFreq、Swapped、NoLeakage、FreqTransitDuet；
10. 输出一份 frequency diagnostics report。

### MVP 不做

1. 不做真实 terminal dispatch retiming；
2. 不做 tick-level HFT；
3. 不做 learnable wavelet；
4. 不做 PINN；
5. 不改动原始 `transit_duet/` 主目录。

### MVP 成功标准

满足以下任意三项即可进入下一阶段：

```text
FreqTransitDuet average wait improves over HRL-Raw / RawHistory
FreqTransitDuet OOD noise setting more stable
Swapped significantly worse than FreqTransitDuet
NoLeakage has higher LowerLFDrift
Promotion reduces recovery time after persistent shift
FocusScore is positive and higher than AllFreq-AllLayers
```

---

## 20. Definition of Done

一个实现只有同时满足下面条件，才可以称为 Freq-HRL，而不是 frequency feature engineering。

```text
[ ] Encoder is causal.
[ ] Upper does not receive raw HF sequence.
[ ] Lower does not receive full LF forecast.
[ ] Upper action is interpretable as low-frequency plan.
[ ] Lower action is interpretable as high-frequency residual correction.
[ ] Persistent HF shock can trigger promotion.
[ ] Lower cumulative action effect is regularized.
[ ] Upper high-frequency action effect is regularized.
[ ] Baselines include RawHistory, AllFreq-AllLayers, Swapped, NoPromotion, NoLeakage.
[ ] Diagnostics include UpperHFPower, LowerLFDrift, FocusScore, PromotionDelay.
[ ] Transit and Quant share core interfaces and differ only in domain adapters and action-effect operators.
```

若以上任意关键项缺失，论文中不能使用 “Frequency-Separated HRL” 作为主方法名称，只能称为 “frequency-enhanced HRL baseline”。

---

## 21. 最终方法摘要

最终实现应满足：

```text
外生信号 x_t 被因果分解为 x^L, x^M, x^H。
高层只用 x^L、低频 forecast、不确定性、高频摘要和 promotion/leakage feedback 输出低通计划。
低层只用当前高层计划、局部 x^H、x^M 和执行误差输出快速 residual correction。
持续高频 residual 通过 promotion gate 升格为新的低频 regime。
低层动作的累计影响通过 action-effect leakage regularization 被阻止长期篡改高层计划。
同一套接口同时实例化为 FreqTransitDuet 和 FreqTradeDuet。
```

这是从“TransitDuet 中分频有效”走向“通用高低频 HRL 控制算法”的最短、最清晰、理论创新最高的路线。
