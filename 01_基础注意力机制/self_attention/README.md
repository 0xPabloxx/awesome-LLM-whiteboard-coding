# Self-Attention (自注意力)

## 概述

自注意力（Self-Attention）是Transformer架构的核心机制，也是现代大语言模型（LLM）的基础。它允许序列中的每个位置都能关注序列中的所有位置（包括自己），从而捕获序列内部的依赖关系。

由Vaswani等人在2017年的论文"Attention is All You Need"中提出，彻底改变了自然语言处理领域。

## 核心思想

自注意力的关键特点：

1. **Query、Key、Value都来自同一个输入**
2. **每个位置关注所有位置**：计算当前位置与序列中所有位置的相关性
3. **并行计算**：不依赖递归，可以完全并行
4. **捕获长距离依赖**：理论上可以关注任意距离的位置

## 数学公式

给定输入序列 $X \in \mathbb{R}^{n \times d}$：

$$
\begin{aligned}
Q &= XW_Q, \quad K = XW_K, \quad V = XW_V \\
\text{Attention}(Q,K,V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{aligned}
$$

其中：
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ 是可学习的投影矩阵
- $\sqrt{d_k}$ 是缩放因子，防止点积过大
- 输出维度与输入相同：$\mathbb{R}^{n \times d}$

## 详细步骤

### 1. 线性投影
```
Q = X @ W_Q  # (seq_len, d_k)
K = X @ W_K  # (seq_len, d_k)
V = X @ W_V  # (seq_len, d_v)
```

### 2. 计算注意力得分
```
scores = Q @ K^T / sqrt(d_k)  # (seq_len, seq_len)
```

### 3. Softmax归一化
```
attention_weights = softmax(scores)  # (seq_len, seq_len)
```

### 4. 加权求和
```
output = attention_weights @ V  # (seq_len, d_v)
```

## Mask机制

### 因果Mask（Causal Mask）
用于GPT等自回归模型，确保当前位置只能看到之前的位置：

```python
mask = np.tril(np.ones((seq_len, seq_len)))
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

### Padding Mask
屏蔽padding位置，避免关注无效token。

## 代码实现

### 基础实现

```python
class SelfAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, x, mask=None):
        # 投影
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 注意力得分
        scores = Q @ K.T / np.sqrt(self.embed_dim)

        # 应用mask
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax + 加权求和
        weights = softmax(scores, axis=-1)
        output = weights @ V

        return output, weights
```

## 特点分析

### 优点
- ✅ **捕获长距离依赖**：任意距离的token可以直接交互
- ✅ **并行计算**：不需要递归，训练速度快
- ✅ **灵活性强**：通过mask可以控制注意力范围
- ✅ **可解释性**：注意力权重可以可视化

### 缺点
- ❌ **计算复杂度高**：时间和空间复杂度都是O(n²)
- ❌ **内存消耗大**：需要存储n×n的注意力矩阵
- ❌ **对长序列不友好**：序列长度翻倍，计算量翻四倍

## 应用场景

### 1. BERT（双向编码器）
- 无mask，每个位置可以看到所有位置
- 用于文本理解任务

### 2. GPT（单向解码器）
- 使用因果mask
- 用于文本生成任务

### 3. Transformer Encoder-Decoder
- Encoder使用双向自注意力
- Decoder使用因果自注意力

## 复杂度分析

| 操作 | 复杂度 |
|------|--------|
| Q、K、V投影 | O(n·d²) |
| 计算QK^T | O(n²·d) |
| Softmax | O(n²) |
| 加权求和 | O(n²·d) |
| **总计** | **O(n²·d + n·d²)** |

当序列长度n较大时，主导项是O(n²·d)。

## 与其他注意力的对比

| 类型 | Query来源 | Key/Value来源 | 用途 |
|------|-----------|---------------|------|
| **Self-Attention** | 输入序列 | 输入序列 | 序列内部依赖 |
| **Cross-Attention** | 解码器 | 编码器 | 跨序列关联 |
| **Soft Attention** | 外部 | 输入序列 | 简单加权 |

## 改进变体

为了解决O(n²)复杂度问题，产生了很多改进：

1. **Linear Attention**：降到O(n)
2. **Sparse Attention**：只计算部分位置
3. **Flash Attention**：优化内存访问
4. **Multi-Query Attention**：共享K、V减少计算

## 使用示例

```python
# 创建自注意力层
self_attn = SelfAttention(embed_dim=512)

# 输入序列
x = np.random.randn(10, 512)  # (seq_len=10, embed_dim=512)

# 标准自注意力
output, weights = self_attn.forward(x)

# 带因果mask（GPT风格）
mask = np.tril(np.ones((10, 10)))
output, weights = self_attn.forward(x, mask=mask)

print(f"输出: {output.shape}")  # (10, 512)
print(f"权重: {weights.shape}")  # (10, 10)
```

## 参考文献

1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers. NAACL 2019.
3. Radford, A., et al. (2018). Improving language understanding by generative pre-training. (GPT)

## 文件说明

- `self_attention.py`: Python实现（带详细中文注释）
- `self_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档
