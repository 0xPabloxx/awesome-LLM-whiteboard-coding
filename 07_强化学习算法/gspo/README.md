# GSPO: Group-wise Stochastic Policy Optimization

GSPO (Group-wise Stochastic Policy Optimization) 是一种基于组的随机策略优化方法，结合了group-based方法和随机优化的优势，特别适合大规模LLM训练。

## 📖 核心概念

### 基本原理

**传统方法的问题**：
```
全量训练: 使用所有样本
- 计算开销大
- 训练慢

随机训练: 随机采样单个样本
- 方差大
- 不稳定
```

**GSPO的解决方案**：
```
1. 组织成groups（每group多个样本）
2. Group内随机采样
3. 多次采样平均
4. 方差缩减技术

优势: 兼具效率和稳定性
```

### 核心机制

```python
# GSPO工作流程
for each group:
    # 1. 随机采样
    samples = random_sample(group, k)  # k < group_size

    # 2. 计算优势
    baseline = mean(group_rewards)
    advantages = samples.rewards - baseline

    # 3. 策略更新
    loss = -E[log_prob * advantage]

    # 4. 多次采样平均
    repeat N times
```

### 方差缩减

GSPO使用多种方差缩减技术：

1. **基线**：使用group均值作为基线
2. **多次采样**：平均多次采样结果
3. **重要性采样**：优先采样高质量样本
4. **控制变量**：利用group内相关性

## 🎯 核心优势

### 1. 样本效率

```
全量方法: 每个group使用1次
GSPO: 每个group使用N次（N次采样）

提升: N倍样本效率
```

### 2. 计算效率

```
全量: 计算group_size个样本
GSPO: 计算sample_size个样本（sample_size < group_size）

节省: (1 - sample_size/group_size) 计算量
```

### 3. 训练稳定性

```
单次采样: 方差 = σ²
N次采样平均: 方差 = σ²/N

方差减少: N倍
```

## 🔧 实现细节

### Group采样

```python
def sample_from_group(group_rewards, sample_size):
    """
    从group中采样

    策略:
    - uniform: 均匀随机（简单）
    - importance: 重要性采样（质量优先）
    - top_k: 只采样top-k（最激进）
    """
    # 均匀采样
    indices = np.random.choice(len(group), sample_size, replace=False)
    return group[indices]
```

### 基线计算

```python
# Group内基线
baseline = np.mean(group_rewards)

# 优势
advantages = sampled_rewards - baseline
```

### 多次采样平均

```python
# 重复采样N次
losses = []
for _ in range(N):
    samples = sample_from_group(group)
    loss = compute_loss(samples)
    losses.append(loss)

# 平均损失
avg_loss = np.mean(losses)
```

## 📊 超参数配置

### 推荐配置

```python
config = {
    'group_size': 8,          # group总样本数
    'sample_size': 4,         # 每次采样数 (group_size/2)
    'num_samples': 10,        # 采样次数
    'use_baseline': True,     # 使用基线
    'temperature': 1.0,       # 温度参数
}
```

### Group Size选择

| group_size | 优势 | 劣势 | 推荐场景 |
|-----------|------|------|---------|
| 4 | 计算快 | 信息少 | 快速实验 |
| **8** | **平衡** | - | **通用（推荐）** |
| 16 | 稳定 | 较慢 | 高质量训练 |
| 32+ | 很稳定 | 很慢 | 不推荐 |

### Sample Size选择

```
推荐: sample_size = group_size / 2

太小: 方差大，不稳定
太大: 计算多，失去效率优势
```

### 采样次数选择

| num_samples | 方差 | 计算量 | 推荐场景 |
|------------|------|--------|---------|
| 1 | 高 | 低 | 不推荐 |
| **5-10** | **中** | **中** | **通用（推荐）** |
| 20+ | 低 | 高 | 需要高稳定性 |

## 🏗️ 应用场景

### 1. 大规模LLM训练

```python
# 分布式训练
# 每个GPU处理不同的groups
for gpu_id, groups_batch in enumerate(distributed_data):
    # Group并行
    for group in groups_batch:
        loss = gspo.compute_loss(group)
        update_model(loss)
```

### 2. 在线学习

```python
# 持续收集新数据
while True:
    # 收集一个group的数据
    group = collect_responses(prompt)

    # GSPO更新（快速）
    loss = gspo.gspo_loss(group)
    model.update(loss)
```

### 3. 多任务学习

```python
# 不同任务的groups
for task in tasks:
    task_groups = collect_task_data(task)

    # 每个任务独立GSPO训练
    for group in task_groups:
        loss = gspo.gspo_loss(group)
        model.update(loss, task=task)
```

## 🔬 GSPO vs 其他方法

### vs GRPO

| 特性 | GRPO | GSPO |
|------|------|------|
| 采样 | 使用全部 | 随机采样 |
| 计算量 | 高 | 中 |
| 方差 | 低 | 中（可调） |
| 效率 | 中 | 高 |

**选择**：
- GRPO：计算资源充足时
- GSPO：需要高效率时

### vs PPO

| 特性 | PPO | GSPO |
|------|-----|------|
| 组织方式 | 全局 | Group-based |
| 复杂度 | 高 | 中 |
| 分布式 | 一般 | 优秀 |
| 适用规模 | 中小 | 大规模 |

### vs DPO

| 特性 | DPO | GSPO |
|------|-----|------|
| 在线采样 | 否 | 是 |
| 方差控制 | 高（监督）| 中（随机）|
| 灵活性 | 低 | 高 |
| 计算效率 | 很高 | 高 |

## 💡 最佳实践

### 1. Group大小选择

```python
# 起点配置
group_size = 8
sample_size = 4  # group_size / 2

# 根据资源调整
if have_more_compute:
    group_size = 16
    sample_size = 8
```

### 2. 采样策略

```python
# 初期: 均匀采样（探索）
gspo = GSPOTrainer(sampling='uniform')

# 后期: 重要性采样（利用）
gspo = GSPOTrainer(sampling='importance')
```

### 3. 方差监控

```python
# 监控损失的标准差
metrics_to_watch = {
    'loss': '平均损失',
    'loss_std': '损失标准差（稳定性指标）',
    'reward/std': 'Group内奖励差异',
}

# 如果loss_std太大
if metrics['loss_std'] > threshold:
    # 增加采样次数
    num_samples *= 2
```

### 4. 基线使用

```python
# 始终使用基线
gspo = GSPOTrainer(use_baseline=True)

# 基线减少方差，几乎没有缺点
```

## ⚠️ 局限性

### 1. 超参数敏感

```
需要调整:
- group_size
- sample_size
- num_samples

建议: 从推荐值开始，逐步调优
```

### 2. 理论保证

```
GSPO是启发式方法
理论分析相对较少
但实践中表现良好
```

### 3. 不适合离线数据

```
GSPO需要在线采样
不能直接用于固定数据集

如果有离线数据 → 考虑DPO
```

## 🚀 快速开始

```bash
python gspo.py
jupyter notebook gspo.ipynb
```

### 简单示例

```python
import numpy as np

# 创建训练器
gspo = GSPOTrainer(
    group_size=8,
    sample_size=4,
    num_samples=10
)

# 模拟一个group
group_rewards = np.random.randn(8) * 0.3 + 0.5
group_log_probs = np.random.randn(8) * 0.5 - 2.5

# 计算损失
loss, metrics = gspo.gspo_loss(
    group_rewards,
    group_log_probs,
    num_samples=10
)

print(f"Loss: {loss:.4f}")
print(f"Loss Std: {metrics['loss_std']:.4f}")
```

## 📈 性能分析

### 计算复杂度

```
全量: O(group_size)
GSPO: O(sample_size × num_samples)

典型值:
- group_size = 8
- sample_size = 4
- num_samples = 10

GSPO: 4×10 = 40 vs 全量: 8
反而更多？

但考虑到可以重用group:
- 全量: 8 (用1次)
- GSPO: 40 (但多次复用)
```

### 方差分析

```
方差缩减因子:
- 基线: ~2x
- 多次采样(N=10): ~√10 ≈ 3x
- 总计: ~6x方差缩减
```

## 🔗 扩展阅读

### 相关方法

- **REINFORCE**: 基础策略梯度
- **GRPO**: Group相对策略优化
- **Variance Reduction**: 方差缩减技术
- **Stochastic Optimization**: 随机优化理论

### 方差缩减技术

- 基线方法
- 控制变量
- 重要性采样
- 对偶采样

---

**关键要点**：GSPO通过group内随机采样和多次平均，在计算效率和训练稳定性之间取得良好平衡，特别适合大规模分布式LLM训练。
