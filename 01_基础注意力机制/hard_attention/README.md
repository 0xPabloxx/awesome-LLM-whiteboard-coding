# Hard Attention (硬注意力)

## 概述

硬注意力（Hard Attention）是一种离散的注意力机制，与软注意力不同，它不对所有位置进行加权求和，而是**只选择部分位置**（通常是1个）进行关注。由于是离散选择，硬注意力不可微分，需要使用强化学习方法（如REINFORCE算法）进行训练。

## 核心思想

硬注意力的核心是**随机采样**或**贪婪选择**：

1. **计算注意力概率**：对所有位置计算注意力得分并归一化
2. **采样位置**：根据概率分布随机采样（训练）或选择最大概率位置（推理）
3. **选择输出**：只使用选中位置的值作为输出

## 数学公式

给定查询 $q$ 和键值对 $(k_1, v_1), ..., (k_n, v_n)$：

$$
\begin{aligned}
e_i &= \text{score}(q, k_i) \\
p_i &= \frac{\exp(e_i)}{\sum_{j=1}^{n} \exp(e_j)} \\
s &\sim \text{Categorical}(p_1, ..., p_n) \\
c &= v_s
\end{aligned}
$$

其中：
- $e_i$ 是注意力得分
- $p_i$ 是注意力概率
- $s$ 是采样得到的位置索引
- $c$ 是输出（只来自位置 $s$）

## 训练方法

由于硬注意力不可微，需要使用**REINFORCE算法**：

$$
\nabla_\theta \mathbb{E}_{s \sim p}[R] = \mathbb{E}_{s \sim p}[R \cdot \nabla_\theta \log p_s]
$$

其中 $R$ 是奖励函数（通常是任务loss的负数）。

## 特点

### 优点
- ✅ **计算高效**：只处理选中的位置，节省计算
- ✅ **真正的注意力聚焦**：强制模型做出明确选择
- ✅ **可解释性强**：明确知道模型在看哪里
- ✅ **适合某些任务**：如需要精确定位的视觉任务

### 缺点
- ❌ **不可微分**：需要强化学习，训练复杂
- ❌ **高方差**：梯度估计方差大，训练不稳定
- ❌ **可能丢失信息**：忽略未选中位置的信息

## 应用场景

1. **图像描述生成**：精确关注图像中的特定物体
2. **视觉问答**：定位问题相关的图像区域
3. **目标检测**：选择性地关注候选区域
4. **文档阅读**：只读取相关的段落

## 代码实现

### 基础实现

```python
class HardAttention:
    def __init__(self, sample_method='stochastic'):
        self.sample_method = sample_method

    def forward(self, query, keys, values):
        # 计算概率
        scores = np.dot(keys, query)
        probs = softmax(scores)

        # 采样位置
        if self.sample_method == 'stochastic':
            location = np.random.choice(len(probs), p=probs)
        else:  # greedy
            location = np.argmax(probs)

        # 只返回选中位置的值
        output = values[location]

        return output, location
```

### Top-K变体

选择多个位置而不是只选1个：

```python
class HardAttentionTopK:
    def __init__(self, k=3):
        self.k = k

    def forward(self, query, keys, values):
        scores = np.dot(keys, query)
        locations = np.argsort(scores)[-self.k:]
        output = np.mean(values[locations], axis=0)
        return output, locations
```

## 使用示例

```python
# 随机采样（训练时）
attention = HardAttention(sample_method='stochastic')
output, location = attention.forward(query, keys, values)
print(f"选中位置: {location}")

# 贪婪选择（推理时）
attention = HardAttention(sample_method='greedy')
output, location = attention.forward(query, keys, values)
print(f"选中位置: {location}")

# Top-K选择
attention = HardAttentionTopK(k=3)
output, locations = attention.forward(query, keys, values)
print(f"选中的3个位置: {locations}")
```

## 与Soft Attention的对比

| 特性 | Soft Attention | Hard Attention |
|------|---------------|----------------|
| **输出方式** | 加权求和所有位置 | 只选择部分位置 |
| **可微性** | ✅ 可微分 | ❌ 不可微分 |
| **训练方法** | 反向传播 | 强化学习 |
| **计算复杂度** | O(n) | O(1) (选1个位置) |
| **训练稳定性** | 稳定 | 不稳定（高方差） |
| **信息利用** | 利用所有信息 | 可能丢失信息 |
| **可解释性** | 权重分散 | 明确选择 |

## 实际应用建议

1. **什么时候用硬注意力？**
   - 任务需要精确定位（如目标检测）
   - 计算资源受限
   - 需要明确的可解释性

2. **什么时候用软注意力？**
   - 需要考虑全局信息
   - 训练稳定性重要
   - 端到端可微分训练

3. **混合方法**
   - 训练时用软注意力（稳定）
   - 推理时用硬注意力（高效）

## 参考文献

1. Xu, K., et al. (2015). Show, attend and tell: Neural image caption generation with visual attention. ICML.
2. Mnih, V., Heess, N., & Graves, A. (2014). Recurrent models of visual attention. NIPS.
3. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning.

## 文件说明

- `hard_attention.py`: Python实现（带详细中文注释）
- `hard_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档
