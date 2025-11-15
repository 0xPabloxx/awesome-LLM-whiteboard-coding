# Soft Attention (软注意力)

## 概述

软注意力（Soft Attention）是最基础的注意力机制形式，由Bahdanau等人在2014年提出，用于改进神经机器翻译模型。它对所有输入位置计算注意力权重，整个过程完全可微分，可以通过反向传播进行端到端训练。

## 核心思想

软注意力机制的核心是**加权求和**：

1. **计算相似度得分**：计算查询(query)与所有键(key)的相似度
2. **归一化权重**：使用softmax将得分转换为概率分布
3. **加权求和**：用权重对值(value)进行加权组合

## 数学公式

给定查询向量 $q$ 和键值对序列 $(k_1, v_1), ..., (k_n, v_n)$：

$$
\begin{aligned}
e_i &= \text{score}(q, k_i) \\
\alpha_i &= \frac{\exp(e_i)}{\sum_{j=1}^{n} \exp(e_j)} \\
c &= \sum_{i=1}^{n} \alpha_i v_i
\end{aligned}
$$

其中：
- $e_i$ 是注意力得分
- $\alpha_i$ 是归一化后的注意力权重（满足 $\sum_i \alpha_i = 1$）
- $c$ 是最终的上下文向量

## 特点

### 优点
- ✅ **可微分**：整个过程可微，可以用梯度下降优化
- ✅ **考虑全局信息**：对所有位置都有关注，不会丢失信息
- ✅ **可解释性**：注意力权重可以可视化，便于理解模型关注的位置

### 缺点
- ❌ **计算复杂度**：需要对所有位置计算注意力，复杂度为O(n)
- ❌ **平滑问题**：由于使用softmax，权重分布可能过于平滑，不够sharp

## 应用场景

1. **机器翻译**：Seq2seq模型中的对齐机制
2. **图像描述生成**：关注图像的不同区域生成文字描述
3. **文本摘要**：选择重要的句子或片段
4. **语音识别**：对齐音频帧和文本

## 代码实现

### 基础实现

```python
def soft_attention_simple(query, keys, values):
    # 计算注意力得分
    scores = np.dot(keys, query)

    # Softmax归一化
    weights = softmax(scores)

    # 加权求和
    output = np.dot(weights, values)

    return output, weights
```

### 参数化实现

使用可学习的参数来计算注意力得分：

```python
class SoftAttention:
    def __init__(self, hidden_dim):
        self.W_a = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_c = np.random.randn(hidden_dim, 1) * 0.01
        self.b_a = np.zeros((hidden_dim,))

    def forward(self, query, keys, values):
        # 计算注意力得分
        scores = []
        for key in keys:
            combined = query + key
            hidden = np.tanh(np.dot(combined, self.W_a) + self.b_a)
            score = np.dot(hidden, self.W_c.flatten())
            scores.append(score)

        # Softmax + 加权求和
        weights = softmax(np.array(scores))
        context = np.dot(weights, values)

        return context, weights
```

## 使用示例

```python
# 创建注意力层
attention = SoftAttention(hidden_dim=64)

# 准备数据
query = np.random.randn(64)
keys = np.random.randn(10, 64)
values = np.random.randn(10, 64)

# 计算注意力
context, weights = attention.forward(query, keys, values)

print(f"上下文向量: {context.shape}")  # (64,)
print(f"注意力权重: {weights.shape}")  # (10,)
print(f"权重总和: {weights.sum()}")     # 1.0
```

## 与Hard Attention的对比

| 特性 | Soft Attention | Hard Attention |
|------|---------------|----------------|
| 可微分 | ✅ 是 | ❌ 否 |
| 训练方法 | 梯度下降 | 强化学习 |
| 计算位置 | 所有位置 | 部分位置 |
| 权重性质 | 连续概率分布 | 离散选择 |
| 计算复杂度 | O(n) | O(1) |

## 参考文献

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. ICLR 2015.
2. Xu, K., et al. (2015). Show, attend and tell: Neural image caption generation with visual attention. ICML 2015.

## 文件说明

- `soft_attention.py`: Python实现（带详细中文注释）
- `soft_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档
