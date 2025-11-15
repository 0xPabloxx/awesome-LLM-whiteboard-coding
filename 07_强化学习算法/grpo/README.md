# GRPO: Group Relative Policy Optimization

GRPO (Group Relative Policy Optimization) 是DeepSeek提出的强化学习算法，专门用于大语言模型的对齐训练。核心创新是在一个group（同一prompt的多个响应）内部进行相对比较和优化，无需显式的奖励模型或参考模型。

## 📖 核心概念

### 基本原理

传统RLHF的问题：
- 需要训练奖励模型（计算开销大）
- 需要参考模型（显存占用多）
- 奖励尺度问题（不同任务奖励范围不同）

GRPO的解决方案：
- ✅ **Group内比较**：对每个prompt采样多个响应，在group内相对排序
- ✅ **相对优势**：使用相对优势而非绝对奖励
- ✅ **自适应基线**：group均值自动作为基线
- ✅ **简化pipeline**：无需奖励模型和参考模型

### GRPO损失函数

```
L_GRPO = -E[log π_θ(y|x) · A_relative(y)]

其中：
A_relative(y) = r(x, y) - baseline
baseline = mean(r(x, y')) for y' in group
```

### 工作流程

```
1. 对每个prompt采样K个响应（形成group）
   prompt x → [y1, y2, y3, y4]

2. 评分（可用简单规则或轻量模型）
   scores → [0.8, 0.5, 0.9, 0.3]

3. 计算相对优势
   baseline = 0.625
   advantages → [0.175, -0.125, 0.275, -0.325]

4. 更新策略
   loss = -E[log π(y) · advantage(y)]
```

## 🎯 核心优势

### 1. 简化Pipeline

| 方法 | 步骤 | 需要模型数 |
|------|------|-----------|
| RLHF (PPO) | SFT → RM → PPO | 3 (策略+奖励+价值) |
| DPO | SFT → DPO | 2 (策略+参考) |
| **GRPO** | **SFT → GRPO** | **1 (策略)** |

### 2. 相对比较更稳定

**绝对奖励的问题**：
```
任务A奖励: [0.1, 0.2, 0.3]  →  需要学习绝对尺度
任务B奖励: [10, 20, 30]     →  尺度不同
```

**相对优势**：
```
任务A优势: [-0.1, 0, 0.1]   →  尺度统一
任务B优势: [-10, 0, 10]     →  相对关系一致
```

### 3. 自适应基线

```python
# 传统方法：需要手动设置基线
baseline = 0.5  # 固定值，不适应不同难度

# GRPO：自动适应
baseline = mean(group_rewards)  # 随任务难度调整
```

## 🔧 实现细节

### Group优势计算

```python
def compute_group_advantages(rewards, method='mean_baseline'):
    """
    计算group内相对优势

    Args:
        rewards: [r1, r2, r3, r4]
        method: 基线方法

    Returns:
        advantages: [A1, A2, A3, A4]
    """
    if method == 'mean_baseline':
        baseline = np.mean(rewards)
    elif method == 'min_baseline':
        baseline = np.min(rewards)
    elif method == 'median_baseline':
        baseline = np.median(rewards)

    advantages = rewards - baseline

    # 标准化（可选）
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    return advantages
```

### 基线方法对比

| 方法 | 公式 | 特点 | 适用场景 |
|------|------|------|---------|
| **Mean** | `mean(r)` | 平衡 | 通用（推荐）|
| Min | `min(r)` | 保守 | 避免负奖励过重惩罚 |
| Median | `median(r)` | 鲁棒 | 有异常值时 |

### 温度参数

```python
advantages = advantages / temperature

# temperature = 0.5: 放大优势差异（激进）
# temperature = 1.0: 原始优势（平衡）
# temperature = 2.0: 缩小优势差异（保守）
```

作用类似DPO的β参数。

## 📊 超参数配置

### 推荐配置

```python
config = {
    'group_size': 4,              # 每个prompt的响应数
    'temperature': 1.0,           # 优势缩放温度
    'advantage_normalization': True,  # 是否标准化优势
    'top_k': None,                # 可选：只用top-k响应
    'learning_rate': 1e-6,        # 学习率
    'batch_size': 64,             # 批次中的group数
}
```

### Group Size选择

| Group Size | 优势 | 劣势 | 适用场景 |
|-----------|------|------|---------|
| 2 | 计算快 | 信息少 | 快速实验 |
| **4** | **平衡** | - | **通用（推荐）** |
| 8 | 信息丰富 | 计算慢 | 有充足计算资源 |
| 16+ | 非常稳定 | 很慢 | 不推荐 |

### Top-K过滤

```python
# 只使用quality最高的k个响应
if top_k is not None:
    top_k_indices = np.argsort(rewards)[-top_k:]
    log_probs = log_probs[top_k_indices]
    advantages = advantages[top_k_indices]
```

好处：
- ✅ 过滤低质量样本
- ✅ 提高训练效率
- ✅ 减少噪声影响

## 🏗️ 实际应用

### 评分函数设计

GRPO的关键是设计合适的评分函数：

**代码生成任务**：
```python
def score_code(prompt, response):
    score = 0.0
    # 1. 执行测试用例
    if passes_tests(response):
        score += 1.0
    # 2. 代码质量
    score += code_quality(response) * 0.3
    # 3. 效率
    score += efficiency(response) * 0.2
    return score
```

**对话任务**：
```python
def score_dialogue(prompt, response):
    score = 0.0
    # 1. 长度合理性
    score += length_score(response) * 0.2
    # 2. 流畅度
    score += fluency_score(response) * 0.3
    # 3. 相关性
    score += relevance_score(prompt, response) * 0.5
    return score
```

**数学任务**：
```python
def score_math(prompt, response):
    # 答案正确性
    correct = extract_answer(response) == ground_truth
    return 1.0 if correct else 0.0
```

### DeepSeek案例

```
模型: DeepSeek-V2
任务: 代码生成、数学推理

配置:
  - group_size: 4-8
  - 评分: 测试用例执行结果
  - temperature: 1.0

结果:
  - 训练更稳定（相比PPO）
  - 无需奖励模型（节省计算）
  - 效果媲美RLHF
```

## 🔬 GRPO vs 其他方法

### vs PPO

| 特性 | PPO | GRPO |
|------|-----|------|
| 奖励模型 | 需要 | 不需要（或轻量）|
| 参考模型 | 需要 | 不需要 |
| 价值函数 | 需要 | 不需要 |
| 实现复杂度 | 高 | 中 |
| 训练稳定性 | 高 | 高 |
| 计算开销 | 高 | 中 |

**结论**：GRPO更简单，计算更高效

### vs DPO

| 特性 | DPO | GRPO |
|------|-----|------|
| 数据形式 | 偏好对 | Group响应 |
| 参考模型 | 需要 | 不需要 |
| 在线采样 | 不需要 | 需要 |
| 灵活性 | 低（只能比较）| 高（任意评分）|
| 训练模式 | 离线 | 在线 |

**权衡**：
- DPO：适合有现成偏好数据的场景
- GRPO：适合能定义评分函数的场景

### 综合对比

```
简单度: DPO > GRPO > PPO
灵活性: PPO > GRPO > DPO
计算效率: DPO > GRPO > PPO
效果: PPO ≈ GRPO ≈ DPO (任务相关)
```

## 💡 最佳实践

### 1. Group Size选择

```python
# 起点：group_size=4
grpo = GRPOTrainer(group_size=4)

# 调整建议：
# - 计算资源充足 → 增大到8
# - 计算资源受限 → 保持4
# - 快速实验 → 降到2
```

### 2. 评分函数设计

```python
# 好的评分函数特点：
✓ 明确：清晰的评分标准
✓ 稳定：同一响应评分一致
✓ 区分度：不同质量有明显差异
✓ 高效：计算不能太慢

# 示例：
def good_score_function(prompt, response):
    # 多个维度组合
    score = 0.0
    score += correctness(response) * 0.6  # 主要
    score += fluency(response) * 0.2      # 次要
    score += brevity(response) * 0.2      # 次要
    return score
```

### 3. 训练监控

```python
metrics_to_watch = {
    'reward/mean': '平均奖励，应该上升',
    'reward/std': '奖励标准差，反映group内差异',
    'advantage/positive_ratio': '正优势比例，接近0.5为佳',
    'loss': '损失，应该下降',
}
```

### 4. 常见问题

**问题1：所有响应质量都很差**
```
现象: 所有rewards都很低
原因: 模型能力不足或评分函数太严格
解决:
  - 降低评分标准
  - 增加更多SFT训练
  - 使用更强的基础模型
```

**问题2：Group内差异太小**
```
现象: reward_std很小，所有rewards接近
原因: 采样多样性不足或评分函数不敏感
解决:
  - 增大采样温度（更多样化）
  - 改进评分函数（增加区分度）
  - 增大group_size
```

**问题3：训练不稳定**
```
现象: 损失波动大
原因: 评分噪声或学习率太大
解决:
  - 降低学习率
  - 增大batch_size
  - 使用advantage_normalization
```

## ⚠️ 局限性

### 1. 需要在线采样

```
GRPO: 需要实时生成响应（计算开销）
DPO: 可以使用离线数据（更高效）

权衡: GRPO更灵活，但计算成本更高
```

### 2. 依赖评分函数质量

```
好的评分函数 → 好的训练效果
差的评分函数 → 差的训练效果

挑战: 并非所有任务都容易设计评分函数
```

### 3. Group Size限制

```
太小: 信息不足，训练不稳定
太大: 计算开销大，效率低

平衡点: 通常4-8是合理范围
```

## 🚀 快速开始

```bash
# 运行示例
python grpo.py

# 运行notebook
jupyter notebook grpo.ipynb
```

### 简单示例

```python
import numpy as np

# 创建GRPO训练器
grpo = GRPOTrainer(group_size=4, temperature=1.0)

# 模拟一个group
rewards = np.array([0.8, 0.5, 0.9, 0.3])
log_probs = np.array([-2.5, -3.0, -2.3, -3.5])

# 计算损失
loss, metrics = grpo.grpo_loss(log_probs, rewards)

print(f"Loss: {loss:.4f}")
print(f"Metrics: {metrics}")
```

## 🔗 扩展阅读

### 论文
- DeepSeek技术报告（GRPO首次提出）
- Group Relative Policy Optimization相关研究

### 相关算法
- **PPO**: 传统RLHF标准算法
- **DPO**: 无需奖励模型的对齐
- **RLOO**: 另一种group-based方法
- **REINFORCE**: 基础策略梯度算法

---

**关键要点**：GRPO通过group内相对比较简化了LLM对齐流程，无需奖励模型和参考模型，在DeepSeek等模型中得到验证，适合有明确评价指标的任务。
