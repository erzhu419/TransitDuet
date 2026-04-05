# TransitDuet：公交时刻表与动态控制协同优化 — 工程开发文档 v3

> **版本历史**：v1 误用 ApproPO 为 action clip，v2 误用 RoboDuet 的共享 encoder + 交叉优势到 single-agent env。
> 本 v3 基于对论文、代码和环境的彻底审计，消除所有问题。

---

## 0. 问题定位与核心贡献

**问题**：公交调度的 timetable（低频/planning）与 holding control（高频/control）在时间尺度上不匹配，传统方法分开处理，无法反馈执行层的困难给计划层。

**TransitDuet 的三个核心贡献**：

1. **ApproPO-Inspired Measurement Projection 保证 Fleet Size 硬约束**
   - 用 OGD 在极锥上更新 θ，自适应调制上层 reward 中各约束分量的权重
   - 当 fleet 长期超标 → θ 自动增大 fleet penalty 权重 → 上层放宽间隔

2. **Lagrangian Soft Constraint 约束 Headway 均匀性**
   - 下层 cost = headway 与 target 的偏差，通过可学习 λ 実现弹性约束

3. **Temporal Advantage Propagation (TAP) 实现异频跨层协同** ← 最核心创新
   - RoboDuet 的交叉优势 `loss = -(A_self + β·A_other)·ratio` 仅适用于同步双头架构
   - TAP 将交叉优势沿**时间轴**展开：上层 trip $i$ 的 advantage 中注入下层在 trip $i$ 生命周期内的 cumulative advantage
   - 这是 RoboDuet 交叉优势在**异步异频层级系统**中的首次推广

---

## 1. 论文正确理解

### 1.1 ApproPO（Miryoosefi et al., 2019）

**核心算法（Algorithm 2）**：
```
for t = 1 to T:
    θ_t ∈ C° ∩ B （极锥 ∩ 单位球）
    π_t ← BEST_RESPONSE(θ_t)    // 用 r = -θ·z 训练完整策略
    ẑ_t ← EST(π_t)              // 估计长期 measurement
    θ_{t+1} ← Γ_Λ(θ_t + η·ẑ_t)  // OGD 更新
return mixture of π_1,...,π_T
```

- **投影对象**：measurement vector $z(\pi)$（策略的长期统计特征），不是 action
- **约束表达**：$C = \{z : z \text{ 满足约束}\}$，通过极锥投影保证 $z(\bar\mu) \to C$
- **输出**：mixture policy（多个策略的均匀混合）

**在 TransitDuet 中的正确用法**：  
不嵌套完整的 BEST_RESPONSE（计算不可行），而是借用 **θ-OGD + measurement projection** 来自适应调制上层 reward 权重。理论 guarantee 降级为 heuristic，但 measurement 约束的精神保留。

### 1.2 RoboDuet（2024）

**核心机制**：
- 共享 actor body + 双 action head（`action_dog_head`, `action_arm_head`）
- 双 critic（`critic_body`, `critic_body_arm`）
- **交叉优势注入**（`ppo.py` L150-154）：`loss_dog = -(A_dog + β·A_arm)·ratio_dog`
- **β annealing**：Stage 1 β=0，Stage 2 β 线性增长到 0.5（`global_switch.py` L62-70）
- **Guidance signal**：arm 输出 `(pitch, roll)` 直接替换 loco 的身体姿态命令

**为什么不能直接迁移到公交 env**：  
RoboDuet 的前提是 dog 和 arm **在同一 timestep 同时执行**，因此共享 body 有物理意义。公交 env 是 single-agent event-driven：holding 在 bus 到站时触发，timetable 在发车时触发，两者完全异步。不存在 `(A_upper, A_lower)` 同时出现在同一个 transition 中的情况。

---

## 2. 环境关键事实（single-agent）

### 2.1 LSTM-RL/env 的 single-agent 模式

```python
# dsac_lag_bus.py L609-651 揭示的模式：
for key in state_dict:              # key = bus_id
    obs = state_dict[key][0]        # 每个 bus 的 obs
    action = policy_net.get_action(obs)  # ← 同一个 policy_net！
    action_dict[key] = action       # 输出给 env
```

**一个策略网络，所有 bus 共享参数**。`action_dict = {bus_id: holding_time}`，但所有 bus 用同一个 `policy_net`。这是 **centralized policy, decentralized execution (parameter sharing)**。

### 2.2 Timetable 的 dispatch 机制

```python
# sim.py L186-189
for i, trip in enumerate(self.timetables):
    if trip.launch_time <= self.current_time and not trip.launched:
        trip.launched = True
        self.launch_bus(trip)
```

Timetable 有 264 个 trip，预定义了 `launch_time`。上层策略介入的正确时机是：**在每个 trip 被 launch 之前，修改该 trip 的 target_headway**。

### 2.3 Reward 中的 360 硬编码

```python
# bus.py L236-237
def headway_reward(headway):
    return -abs(headway - 360)       # ← 硬编码 360！
# 以及 L242: abs(self.forward_headway - 360), L252: abs(self.forward_headway - 360)
```

**所有 headway 评价都以 360s 为 target**。如果上层给 target_headway=480s，下层仍然朝 360s 优化。必须修改为动态读取 target_headway。

### 2.4 bus._prepare_for_action 不接收 sim / timetable

```python
# bus.py L218
def _prepare_for_action(self, current_time, bus_all, debug):
    # ← 没有 sim 对象，无法获取 timetable 的 target_headway
```

需要通过 `bus.drive()` → `_prepare_for_action()` 的调用链传入 target_headway。

---

## 3. 系统架构（v3 最终版）

```
┌───────────────────────────────────────────────────────────────┐
│                     TransitDuet Runner                        │
│                                                               │
│   πU: Upper Policy (独立网络)     πL: Lower Policy (独立网络)  │
│   ┌─────────────────┐            ┌─────────────────┐          │
│   │ s_U: [hour,     │            │ s_bus: [bus_id,  │          │
│   │  demand, fleet, │            │  station, hw,    │          │
│   │  prev_θ_weights]│            │  hw_dev, speed..]│          │
│   │                 │            │                  │          │
│   │ a_U ∈ R^3:      │            │ a_L ∈ [0,60]:    │          │
│   │ [H_peak,H_off,  │            │ holding_time     │          │
│   │  H_transition]  │            │                  │          │
│   │                 │            │ freq: per station │          │
│   │ freq: per       │            │ arrival (~20/ep/  │          │
│   │ dispatch (~132  │            │ bus)              │          │
│   │ per ep)         │            │                  │          │
│   └────────┬────────┘            └────────┬────────┘          │
│            │ target_headway               │ holding           │
│            ▼                              ▼                   │
│   ┌──────────────────────────────────────────────────┐        │
│   │       env_bus (single-agent parameter sharing)   │        │
│   │                                                  │        │
│   │ - 264 trips, launch_bus when time arrives        │        │
│   │ - πU 在 trip launch 前修改 target_headway         │        │
│   │ - πL 在 bus 到站时推理 holding_time              │        │
│   │ - bus.obs 中新增 headway_deviation                │        │
│   │ - bus.cost = (hw - target_hw)² / target_hw²      │        │
│   │ - sim 记录 measurement_vector & 并行车辆数        │        │
│   └──────────────────────────────────────────────────┘        │
│                                                               │
│   ┌─────────────── Coupling Mechanisms ──────────────┐        │
│   │                                                  │        │
│   │  1. Guidance: πU → target_headway → bus.obs      │        │
│   │                                                  │        │
│   │  2. ApproPO θ-OGD: z=[wait,fleet,bunch]          │        │
│   │     → θ 调制 R_upper 各分量权重                    │        │
│   │     → fleet 超标 → θ 增大 fleet penalty 权重      │        │
│   │                                                  │        │
│   │  3. TAP (Temporal Advantage Propagation):        │        │
│   │     A_U[i] += β·mean(A_L[j∈trip_i])             │        │
│   │     A_L[j] += β·A_U[trip_of(j)]                 │        │
│   │                                                  │        │
│   │  4. Lagrangian: λ·(E[cost] - c_limit)           │        │
│   │     cost = headway deviation from target         │        │
│   └──────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────┘
```

### 3.1 为什么两个独立策略而非共享 encoder

| RoboDuet | TransitDuet | 差异原因 |
|----------|-------------|---------|
| Dog + Arm 在同一 timestep 同时执行 | Dispatch + Holding 异步触发 | 公交 env 是 event-driven |
| 共享 body → 同一 hidden → 两个 head | 不可行：两个 head 不在同一时刻被调用 | 输入空间不同（s_U ≠ s_bus） |
| 2-dim reward per transition | 每个 transition 只有一个 reward | 无法拼接 `(r_upper, r_lower)` |

### 3.2 TAP 为什么是正确的跨层协同

RoboDuet 的交叉优势：
```python
loss_dog = -(A_dog + β·A_arm) * ratio_dog  # 同一 timestep 的两个 advantage
```

TAP（Temporal Advantage Propagation）——异步版本：
```python
# 上层 trip i 发出后，下层在 trip i 的所有站控制中积累 advantage
# trip i 的 "生命周期" = 从 dispatch 到 trip i+2 dispatch（同方向下一班）
trip_lower_advantages = [A_L[j] for j in lower_transitions if j.trip_id == i]
A_U_augmented[i] = A_U[i] + beta * mean(trip_lower_advantages)

# 反向：下层 transition j 属于 trip i，注入上层 trip i 的 advantage
A_L_augmented[j] = A_L[j] + beta * A_U[trip_of(j)]
```

**直觉**：如果上层给了一个不合理的 target_headway=180s（太密），下层在执行中会产生大量负 advantage（holding 困难、串车频繁）。TAP 将这些负 advantage 反馈给上层的该 trip transition，使上层学到"这个间隔不好"。

**与 RoboDuet 的精确对应**：
| RoboDuet | TAP |
|---|---|
| `A_dog + β·A_arm` | `A_U[i] + β·mean(A_L[j∈trip_i])` |
| 同一 timestep 的同步注入 | 跨 trip 生命周期的时序聚合 |
| β 线性增长 0→0.5 | 同样的 β annealing |

---

## 4. 环境改造规范

### 4.1 `timetable.py`

```python
class Timetable(object):
    def __init__(self, launch_time, launch_turn, direction, target_headway=360.0):
        self.launch_time = launch_time
        self.direction = direction
        self.launch_turn = launch_turn
        self.launched = False
        self.target_headway = target_headway  # 由上层写入
```

### 4.2 `sim.py` 改造

**4.2.1 dispatch 事件中调用上层策略**：

```python
# sim.py step() 中，L186-189 修改
for i, trip in enumerate(self.timetables):
    if trip.launch_time <= self.current_time and not trip.launched:
        # ── 上层策略介入点 ──
        # 在 launch 前，让 πU 决定 target_headway
        # 这个回调由 runner 注册
        if self._upper_policy_callback is not None:
            s_upper = self._build_upper_state(trip)
            target_hw = self._upper_policy_callback(s_upper, trip)
            trip.target_headway = target_hw
        
        trip.launched = True
        self.launch_bus(trip)
```

**4.2.2 drive() 传入 target_headway**：

```python
# sim.py L218 修改
if bus.on_route:
    target_hw = self._get_target_headway_for_bus(bus)
    bus.drive(self.current_time, action[bus.bus_id], self.bus_all,
              debug=debug, target_headway=target_hw)

def _get_target_headway_for_bus(self, bus):
    for tt in self.timetables:
        if tt.launch_turn == bus.trip_id:
            return tt.target_headway
    return 360.0
```

**4.2.3 step() 返回 cost + 记录 concurrent vehicles**：

```python
# 新增 cost dict
self.cost = {key: 0.0 for key in range(self.max_agent_num)}

# 在 step() 的 reward 收集后面增加：
self.cost_list = [bus for bus in self.bus_all if bus.cost is not None]
for bus in self.cost_list:
    self.cost[bus.bus_id] = bus.cost

# 记录并行车辆数
concurrent = sum(1 for bus in self.bus_all if bus.on_route)
self._peak_concurrent = max(getattr(self, '_peak_concurrent', 0), concurrent)

return self.state, self.reward, self.cost, self.done
```

**4.2.4 上层状态构建**：

```python
def _build_upper_state(self, trip):
    """
    构建上层策略的 state，在每个 dispatch 事件时调用。
    """
    hour = 6 + self.current_time // 3600
    
    # 当前小时的预估客流（从各站 OD 的当前时段聚合）
    effective_period_str = f"{min(hour, 19):02}:00:00"
    total_demand = sum(
        sum(od.get(effective_period_str, {}).values())
        for s in self.stations if s.od is not None
        for od in [s.od]
    )
    
    # 当前在线车辆数
    fleet_on_route = sum(1 for bus in self.bus_all if bus.on_route)
    
    # 已发出但同方向上一 trip 的 actual headway
    same_dir_launched = [tt for tt in self.timetables
                         if tt.launched and tt.direction == trip.direction]
    if len(same_dir_launched) >= 2:
        last_two = sorted(same_dir_launched, key=lambda t: t.launch_time)[-2:]
        prev_actual_headway = last_two[1].launch_time - last_two[0].launch_time
    else:
        prev_actual_headway = 360.0
    
    # 当前串车率（近 10 trip 的 unhealthy 比例）
    recent_trips = [tt for tt in self.timetables if tt.launched][-10:]
    # 简化：用 bus_all 中 is_unhealthy 的比例
    unhealthy_rate = sum(1 for bus in self.bus_all if bus.is_unhealthy) / max(len(self.bus_all), 1)
    
    return np.array([
        hour / 24.0,                          # 归一化时刻
        total_demand / 1000.0,                 # 归一化客流
        fleet_on_route / self.max_agent_num,   # 归一化在线车辆
        prev_actual_headway / 600.0,           # 归一化前次间隔
        unhealthy_rate,                        # 串车率
    ], dtype=np.float32)

@property
def measurement_vector(self):
    """ApproPO 需要的长期 measurement，episode 结束后计算。"""
    # z[0]: 平均等待时间（分钟）
    total_wait, pax_count = 0.0, 0
    for s in self.stations:
        for p in s.total_passenger:
            if hasattr(p, 'boarding_time') and p.boarding_time is not None:
                total_wait += (p.boarding_time - p.appear_time)
                pax_count += 1
    avg_wait = (total_wait / max(pax_count, 1)) / 60.0
    
    # z[1]: 峰值并行车辆
    peak_fleet = getattr(self, '_peak_concurrent', 0)
    
    # z[2]: 串车率
    unhealthy = sum(1 for bus in self.bus_all if bus.is_unhealthy)
    bunching_rate = unhealthy / max(len(self.bus_all), 1)
    
    return np.array([avg_wait, peak_fleet, bunching_rate])
```

### 4.3 `bus.py` 改造

**4.3.1 drive() 接收 target_headway**：

```python
# bus.py drive() 签名修改
def drive(self, current_time, action, bus_all, debug=False, target_headway=360.0):
    self._target_headway = target_headway
    # ... 原逻辑不变
```

**4.3.2 _prepare_for_action() 改造**：

```python
def _prepare_for_action(self, current_time, bus_all, debug):
    self.forward_bus = list(filter(lambda x: self.trip_id - 2 in x.trip_id_list, bus_all))
    self.backward_bus = list(filter(lambda x: self.trip_id + 2 in x.trip_id_list, bus_all))

    if self.next_station in self.effective_station[2:] and (len(self.forward_bus) != 0 or len(self.backward_bus) != 0):
        # ── obs 扩展：增加 headway_deviation ──
        target_hw = getattr(self, '_target_headway', 360.0)
        headway_dev = (self.forward_headway - target_hw) / max(target_hw, 1.0)
        
        self.obs = [
            self.bus_id,
            self.last_station.station_id,
            current_time // 3600,
            self.direction,
            self.forward_headway,
            self.backward_headway,
            len(self.next_station.waiting_passengers) * 1.5 + self.current_route.distance / self.current_route.speed_limit,
            headway_dev,  # ★ 新增：归一化 headway 偏差
        ]
        all_route = self.routes_list[:len(self.routes_list) // 2] if self.direction else self.routes_list[len(self.routes_list) // 2:]
        speed_list = [all_route[i].speed_limit for i in range(len(all_route))]
        self.obs.extend(speed_list)

        # ── reward：使用 target_headway 而非硬编码 360 ──
        def headway_reward(headway):
            return -abs(headway - target_hw)

        forward_reward = headway_reward(self.forward_headway) if len(self.forward_bus) != 0 else None
        backward_reward = headway_reward(self.backward_headway) if len(self.backward_bus) != 0 else None
        if forward_reward is not None and backward_reward is not None:
            weight = abs(self.forward_headway - target_hw) / (abs(self.forward_headway - target_hw) + abs(self.backward_headway - target_hw) + 1e-6)
            similarity_bonus = -abs(self.forward_headway - self.backward_headway) * 0.5
            self.reward = forward_reward * weight + backward_reward * (1 - weight) + similarity_bonus
        elif forward_reward is not None:
            self.reward = forward_reward
        elif backward_reward is not None:
            self.reward = backward_reward
        else:
            self.reward = -50

        if abs(self.forward_headway - target_hw) > 180 or abs(self.backward_headway - target_hw) > 180:
            self.reward -= 20
            self.is_unhealthy = True

        # ── cost：Lagrangian 约束用 ──
        self.cost = min(headway_dev ** 2, 1.0)

    self.state = BusState.WAITING_ACTION
```

> **state_dim 变化**：从 `7 + len(routes)//2` 变为 `8 + len(routes)//2`。

---

## 5. 上层策略 πU

### 5.1 网络与动作空间

```python
class UpperPolicy(nn.Module):
    """上层 timetable 策略。独立网络，不与 πL 共享参数。"""
    def __init__(self, state_dim=5, K=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, K)
        self.log_std_head = nn.Linear(hidden_dim, K)
        
        # 动作范围：[H_peak, H_off_peak, H_transition]
        self.action_low  = torch.tensor([180., 300., 240.])
        self.action_high = torch.tensor([600., 1200., 900.])
    
    def forward(self, state):
        h = self.net(state)
        mean = torch.sigmoid(self.mean_head(h))  # [0,1]
        # 映射到 [action_low, action_high]
        action = self.action_low + mean * (self.action_high - self.action_low)
        log_std = torch.clamp(self.log_std_head(h), -5, 0)
        return action, log_std
    
    def get_action(self, state, deterministic=False):
        action, log_std = self.forward(state)
        if deterministic:
            return action.detach().cpu().numpy()
        std = log_std.exp()
        noise = torch.randn_like(action) * std
        return torch.clamp(action + noise, self.action_low, self.action_high).detach().cpu().numpy()
```

### 5.2 上层 Reward（ApproPO θ 调制）

```python
class MeasurementProjection:
    """
    ApproPO-inspired: 用 θ-OGD 自适应调制 R_upper 中各约束分量的权重。
    """
    def __init__(self, N_fleet=12, d=3, lr=0.01):
        self.N_fleet = N_fleet
        self.d = d
        self.theta = np.zeros(d)
        self.theta[0] = -1.0  # 初始偏重等待时间
        self.lr = lr
        self._iter = 1
    
    def update(self, z_observed):
        """Episode 结束后用观测 measurement 更新 θ"""
        # fleet violation signal
        fleet_violation = max(0, z_observed[1] - self.N_fleet)
        loss_vector = np.array([
            -z_observed[0],       # 等待时间越高 → 越应惩罚
            -fleet_violation,     # fleet 越超标 → 越应惩罚
            -z_observed[2],       # 串车率越高 → 越应惩罚
        ])
        update = (self.lr / np.sqrt(self._iter)) * loss_vector
        self.theta = self.theta + update
        # 投影到单位球（极锥投影的简化版）
        norm = np.linalg.norm(self.theta)
        if norm > 1:
            self.theta /= norm
        self._iter += 1
    
    def compute_upper_reward(self, z):
        """用 θ 加权的上层 reward"""
        fleet_over = max(0, z[1] - self.N_fleet) ** 2
        penalties = np.array([z[0], fleet_over, z[2]])
        return float(np.dot(self.theta, -penalties))
```

### 5.3 上层触发时机

上层策略在**每个 trip dispatch 时**被调用（理解 C），一个 episode 约 132 个上层 transition。

```python
# runner 注册的 callback
def upper_policy_callback(s_upper, trip):
    """sim.step() 在每个 dispatch 事件时调用此函数"""
    with torch.no_grad():
        s_tensor = torch.from_numpy(s_upper).float()
        action = upper_policy.get_action(s_tensor, deterministic=not training)
    
    # 根据 trip 的时段确定使用哪个参数
    hour = 6 + trip.launch_time // 3600
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        target_hw = action[0]  # H_peak
    elif 9 < hour < 17:
        target_hw = action[1]  # H_off_peak
    else:
        target_hw = action[2]  # H_transition
    
    return float(target_hw)
```

---

## 6. 下层策略 πL

### 6.1 DSAC-Lagrangian

基于 `omnisafe/Holding_control/dsac_lag_bus.py`，增加：

```python
class DSAC_Lagrangian_Trainer(DSAC_Trainer):
    def __init__(self, ..., cost_limit=0.15):
        super().__init__(...)
        self.cost_limit = cost_limit
        self.log_lambda = torch.zeros(1, requires_grad=True, device=device)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=1e-3)
    
    @property
    def lambda_param(self):
        return self.log_lambda.exp()

    def update(self, batch_size, ...):
        state, action, reward, cost, next_state, done = self.replay_buffer.sample(batch_size)
        
        # ... 原 DSAC critic + actor 更新 ...
        
        # 策略 loss 增加 Lagrangian 惩罚
        policy_loss = (self.alpha * log_prob
                       - q_new_actions
                       + self.lambda_param.detach() * cost_q_new_actions
                      ).mean()
        
        # Lambda dual 更新
        lambda_loss = -self.lambda_param * (self.cost_limit - cost_q_new_actions.mean().detach())
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.log_lambda.data.clamp_(min=-10.0)
```

### 6.2 CostReplayBuffer

```python
class CostReplayBuffer:
    """支持 (s, a, r, c, s', done, trip_id) 七元组"""
    def push(self, state, action, reward, cost, next_state, done, trip_id): ...
    def sample(self, batch_size): ... → (s, a, r, c, s', done, trip_id)
```

> **trip_id 字段**：TAP 需要知道每个下层 transition 属于哪个 trip。

---

## 7. TAP：Temporal Advantage Propagation

### 7.1 算法描述

在每个 episode 结束后，做 TAP 后处理：

```python
class TAPManager:
    def __init__(self, beta_schedule):
        self.beta_schedule = beta_schedule
        self.upper_transitions = []  # [(s_U, a_U, r_U, s'_U, trip_id)]
        self.lower_advantages = {}   # {trip_id: [A_L values]}
    
    def record_upper_transition(self, s, a, r, s_next, trip_id):
        self.upper_transitions.append((s, a, r, s_next, trip_id))
    
    def record_lower_advantage(self, advantage, trip_id):
        if trip_id not in self.lower_advantages:
            self.lower_advantages[trip_id] = []
        self.lower_advantages[trip_id].append(advantage)
    
    def compute_augmented_advantages(self, episode):
        beta = self.beta_schedule.get_beta(episode)
        
        augmented_upper = []
        for s, a, r, s_next, trip_id in self.upper_transitions:
            A_U = r  # 简化：用 return 近似 advantage
            if trip_id in self.lower_advantages and self.lower_advantages[trip_id]:
                cross_term = beta * np.mean(self.lower_advantages[trip_id])
            else:
                cross_term = 0.0
            augmented_upper.append(A_U + cross_term)
        
        return augmented_upper

class BetaSchedule:
    def __init__(self, warmup_eps=50, ramp_eps=100):
        self.warmup = warmup_eps
        self.ramp = ramp_eps
    
    def get_beta(self, episode):
        if episode <= self.warmup:
            return 0.0
        elif episode < self.warmup + self.ramp:
            return 0.5 * (episode - self.warmup) / self.ramp
        else:
            return 0.5
```

### 7.2 TAP 的物理意义

| 场景 | TAP 如何帮助 |
|------|-------------|
| 上层给了过小的 H_target=180s | 下层执行困难 → A_L 大量为负 → TAP 将负 A_L 注入 A_U → 上层学到"放宽间隔" |
| 上层给了过大的 H_target=1200s | 下层轻松但等待时间长 → R_upper 中 avg_wait 项为大负值 → 上层学到"收紧间隔" |
| 上层给了合理 H_target=360s | A_L 接近 0 → TAP 几乎不修正 → 上层保持当前策略 |

---

## 8. 训练循环

```python
class TransitDuetRunner:
    def __init__(self, config):
        self.env = env_bus(config.env_path, route_sigma=config.route_sigma)
        
        # 两个独立策略
        self.upper_policy = UpperPolicy(state_dim=5, K=3)
        self.lower_trainer = DSAC_Lagrangian_Trainer(
            env=self.env, replay_buffer=CostReplayBuffer(1e6),
            hidden_dim=config.hidden_dim, action_range=60.0,
            cost_limit=config.cost_limit
        )
        
        # ApproPO measurement projection
        self.measurement_proj = MeasurementProjection(N_fleet=config.N_fleet)
        
        # TAP
        self.beta_schedule = BetaSchedule(warmup_eps=50, ramp_eps=100)
        self.tap = TAPManager(self.beta_schedule)
        
        # 上层 optimizer
        self.upper_optimizer = optim.Adam(self.upper_policy.parameters(), lr=1e-4)
    
    def run(self, total_episodes=300):
        for ep in range(total_episodes):
            self.tap.clear()
            self.env.reset()
            
            # 注册上层 callback
            if ep >= self.beta_schedule.warmup:
                self.env._upper_policy_callback = self._upper_callback
            else:
                self.env._upper_policy_callback = None  # warmup: 固定 360s
            
            state_dict, reward_dict, _ = self.env.initialize_state()
            done = False
            action_dict = {k: None for k in range(self.env.max_agent_num)}
            
            while not done:
                # 下层动作（与 dsac_lag_bus.py 相同的 single-agent 模式）
                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            obs = np.array(state_dict[key][0])
                            action_dict[key] = self.lower_trainer.policy_net.get_action(
                                torch.from_numpy(obs).float())
                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            state = np.array(state_dict[key][0])
                            next_state = np.array(state_dict[key][1])
                            reward = reward_dict[key]
                            cost = self.env.cost[key]
                            trip_id = int(state_dict[key][0][0])  # bus_id → trip mapping
                            
                            self.lower_trainer.replay_buffer.push(
                                state, action_dict[key], reward, cost, next_state, done, trip_id)
                            
                            # TAP: 记录下层 advantage（简化为 reward）
                            self.tap.record_lower_advantage(reward, trip_id)
                        
                        state_dict[key] = state_dict[key][1:]
                        obs = np.array(state_dict[key][0])
                        action_dict[key] = self.lower_trainer.policy_net.get_action(
                            torch.from_numpy(obs).float())
                
                state_dict, reward_dict, cost_dict, done = self.env.step(action_dict)
                
                # 下层训练
                if len(self.lower_trainer.replay_buffer) > config.batch_size:
                    self.lower_trainer.update(config.batch_size, ...)
            
            # ── Episode 结束 ──
            
            # ApproPO measurement 更新
            z = self.env.measurement_vector
            self.measurement_proj.update(z)
            
            # 上层 reward
            r_upper = self.measurement_proj.compute_upper_reward(z)
            
            # TAP: 计算增强后的上层 advantage
            augmented_A = self.tap.compute_augmented_advantages(ep)
            
            # 上层策略更新（简化版 policy gradient）
            if ep >= self.beta_schedule.warmup:
                self._update_upper_policy(augmented_A)
```

---

## 9. 文件结构

```
TransitDuet/
├── roadmap.md
├── ApproPO/                        # 参考（不直接调用）
├── RoboDuet/                       # 参考（TAP 来源）
├── reference/                      # 论文 PDFs
│
└── transit_duet/                   # ★ 主代码
    ├── env/                        # 从 LSTM-RL/env 复制改造
    │   ├── bus.py                  # +target_headway, +headway_dev obs, +cost, reward 改 360→target_hw
    │   ├── sim.py                  # +upper_callback, +cost dict, +measurement_vector, +_build_upper_state
    │   ├── timetable.py            # +target_headway 字段
    │   ├── station.py              # 不改
    │   └── route.py, passenger.py  # 不改
    │
    ├── upper/
    │   ├── upper_policy.py         # UpperPolicy (独立 MLP)
    │   └── measurement_proj.py     # MeasurementProjection (ApproPO θ-OGD)
    │
    ├── lower/
    │   ├── dsac_lagrangian.py      # DSAC_Lagrangian_Trainer
    │   └── cost_replay_buffer.py   # CostReplayBuffer (含 trip_id)
    │
    ├── coupling/
    │   ├── tap.py                  # TAPManager (Temporal Advantage Propagation)
    │   └── beta_schedule.py        # BetaSchedule (0→0.5 annealing)
    │
    ├── runner.py                   # TransitDuetRunner
    ├── config.yaml
    └── eval.py
```

---

## 10. 超参数

```yaml
env:
  route_sigma: 1.5
  max_agent_num: 25
  effective_trip_num: 264

upper:
  state_dim: 5               # [hour, demand, fleet, prev_hw, unhealthy_rate]
  action_dim: 3               # [H_peak, H_off_peak, H_transition]
  action_low: [180, 300, 240]
  action_high: [600, 1200, 900]
  hidden_dim: 64
  lr: 1e-4
  N_fleet: 12

lower:
  state_dim: "8 + routes//2"  # 原 7 + headway_dev
  hidden_dim: 32
  action_range: 60.0
  cost_limit: 0.15
  batch_size: 2048
  lr: 1e-5
  lambda_lr: 1e-3

coupling:
  beta_warmup_eps: 50         # Stage 1: 只训练下层
  beta_ramp_eps: 100          # Stage 2: β 0→0.5
  beta_max: 0.5
  measurement_lr: 0.01        # ApproPO θ-OGD 学习率

total_episodes: 300
```

---

## 11. 实验规划

### Phase 1: 下层验证（Stage 1, β=0, ep 0-50）

固定 target_headway=360s，验证 DSAC-Lagrangian 收敛。
- **通过标准**：cost (headway deviation²) 持续下降，reward 上升。

### Phase 2: 联合训练（Stage 2, β ramp, ep 50-150）

上层介入，TAP 生效。
- **关键观察**：peak_fleet 是否被 θ-OGD 控制在 ≤ N_fleet；mean(target_headway) 是否自适应。

### Phase 3: 消融实验

| 组 | 省略 | 验证 |
|---|---|---|
| A. 完整 TransitDuet | — | baseline |
| B. 无 TAP | β=0 永远 | TAP 的价值 |
| C. 无 θ-OGD | 固定 reward 权重 | measurement projection 的价值 |
| D. 无 Lagrangian | cost_limit=∞ | 下层软约束 |
| E. 固定 timetable | πU 不训练 | 传统分离方法 |
| F. 扁平 RL | 单层 DSAC | 层级结构的必要性 |

### Phase 4: 泛化测试

训练 σ=1.5，测试 σ ∈ {0.5, 1.0, 2.0, 3.0}

---

## 12. 创新点总结

1. **Temporal Advantage Propagation (TAP)**：RoboDuet 交叉优势在异步双层系统中的首次推广。RoboDuet 是同步版（同一 timestep 的 A_dog + β·A_arm），TAP 是时序展开版（跨 trip 生命周期的 advantage aggregation）。

2. **ApproPO θ-OGD 用于 Fleet Size 硬约束**：首次在交通调度中用 measurement projection 保证长期统计行为满足约束。不同于 Lagrangian（允许局部违约），θ-OGD 的 reward weight 调制从根源上引导策略远离 fleet 超标区域。

3. **软硬约束分离架构**：上层（fleet → measurement projection）和下层（headway → Lagrangian）通过不同数学机制处理不同约束类型，架构可解释性强，各机制可独立消融。

4. **可扩展到多线路**：架构设计预留了 `line_idx` 维度扩展能力。

---

## 附录 A: 验证 Checklist

- [ ] `bus.drive()` 接收 `target_headway` 参数
- [ ] `headway_reward()` 使用 `self._target_headway` 而非 360
- [ ] `bus.obs` 长度 = `8 + routes//2`（新增 headway_dev）
- [ ] `env.step()` 返回 4 元组 `(state, reward, cost, done)`
- [ ] `env._upper_policy_callback` 在 dispatch 时被调用（每 ep 约 132 次）
- [ ] `env.measurement_vector` 在 ep 结束后返回 `[avg_wait, peak_fleet, bunching_rate]`
- [ ] Phase 1 (50 ep): cost 下降
- [ ] Phase 2 (100 ep): peak_fleet ≤ N_fleet（θ 调制生效）
- [ ] 消融 B (无 TAP) vs A (完整): A 明显优于 B

## 附录 B: 代码依赖

```
LSTM-RL/env/ → transit_duet/env/（复制改造 bus.py, sim.py, timetable.py）
omnisafe/Holding_control/dsac_lag_bus.py → transit_duet/lower/dsac_lagrangian.py（增加 Lagrangian）
RoboDuet/global_switch.py → transit_duet/coupling/beta_schedule.py（β annealing）
RoboDuet/ppo.py L150-154 → transit_duet/coupling/tap.py（从同步 cross-advantage 推广到异步 TAP）
ApproPO/olo.py → transit_duet/upper/measurement_proj.py（θ-OGD 更新）
不修改任何原有仓库的文件。
```