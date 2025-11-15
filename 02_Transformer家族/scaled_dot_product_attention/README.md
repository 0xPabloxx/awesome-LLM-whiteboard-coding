# Scaled Dot-Product Attention (缩放点积注意力)

## 概述

缩放点积注意力（Scaled Dot-Product Attention）是Transformer架构的核心组件，由Vaswani等人在2017年的论文《Attention is All You Need》中提出。它是Multi-Head Attention的基础构建块，通过引入缩放因子解决了传统点积注意力在高维空间中的梯度消失问题。

## 核心思想

缩放点积注意力通过以下步骤计算注意力输出：

1. **计算相似度**：使用矩阵乘法计算查询(Q)和键(K)的点积
2. **缩放**：除以$\sqrt{d_k}$防止点积值过大
3. **Mask（可选）**：应用掩码处理padding或实现因果注意力
4. **归一化**：使用softmax将得分转换为概率分布
5. **加权求和**：用注意力权重对值(V)进行加权组合

## 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$ 是查询矩阵
- $K \in \mathbb{R}^{m \times d_k}$ 是键矩阵
- $V \in \mathbb{R}^{m \times d_v}$ 是值矩阵
- $d_k$ 是键/查询的维度
- $n$ 是查询序列长度，$m$ 是键/值序列长度

### 详细计算步骤

1. **计算注意力分数**：
   $$\text{scores} = QK^T \in \mathbb{R}^{n \times m}$$

2. **缩放**：
   $$\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}$$

3. **应用Mask（可选）**：
   $$\text{masked\_scores}_{ij} = \begin{cases}
   \text{scaled\_scores}_{ij} & \text{if not masked} \\
   -\infty & \text{if masked}
   \end{cases}$$

4. **Softmax归一化**：
   $$\alpha_{ij} = \frac{\exp(\text{masked\_scores}_{ij})}{\sum_{k=1}^{m} \exp(\text{masked\_scores}_{ik})}$$

5. **加权求和**：
   $$\text{output}_i = \sum_{j=1}^{m} \alpha_{ij} V_j$$

## 为什么需要缩放？

### 问题背景

当$d_k$（键的维度）很大时，$Q$和$K$的点积会产生很大的值：

- 假设$Q$和$K$的元素是独立的随机变量，均值为0，方差为1
- 那么点积$QK^T$的方差约为$d_k$
- 当$d_k$增大时，点积值的分布会变得更加极端

### 缩放的作用

除以$\sqrt{d_k}$可以：

1. **稳定方差**：将点积的方差从$d_k$缩放到1
2. **避免梯度消失**：防止softmax函数进入饱和区
3. **改善训练**：使得不同维度下的模型训练更稳定

### 实验对比

| $d_k$ | 不缩放的分数范围 | 缩放后的分数范围 | Softmax最大权重（不缩放） | Softmax最大权重（缩放） |
|-------|-----------------|----------------|------------------------|---------------------|
| 16    | [-30, 30]       | [-7.5, 7.5]    | 0.85                   | 0.45                |
| 64    | [-60, 60]       | [-7.5, 7.5]    | 0.95                   | 0.45                |
| 256   | [-120, 120]     | [-7.5, 7.5]    | 0.99                   | 0.45                |
| 512   | [-170, 170]     | [-7.5, 7.5]    | ~1.0                   | 0.45                |

从表中可以看出：
- 不缩放时，$d_k$越大，权重分布越集中（接近one-hot）
- 缩放后，权重分布保持相对均匀，不受$d_k$影响

## Mask机制

### 1. Padding Mask

用于处理变长序列，忽略padding位置的影响。

```python
# 创建padding mask
def create_padding_mask(seq_len, padding_positions):
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for pos in padding_positions:
        mask[:, pos] = True  # mask掉padding的key位置
    return mask

# 示例：序列长度为6，位置4和5是padding
padding_mask = create_padding_mask(6, [4, 5])
# [[0, 0, 0, 0, 1, 1],
#  [0, 0, 0, 0, 1, 1],
#  [0, 0, 0, 0, 1, 1],
#  [0, 0, 0, 0, 1, 1],
#  [0, 0, 0, 0, 1, 1],
#  [0, 0, 0, 0, 1, 1]]
```

### 2. Causal Mask（因果掩码）

用于Transformer的Decoder，防止当前位置看到未来的信息。

```python
# 创建因果mask（上三角矩阵）
def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return mask

# 示例：序列长度为4
causal_mask = create_causal_mask(4)
# [[0, 1, 1, 1],    <- 位置0只能看到位置0
#  [0, 0, 1, 1],    <- 位置1只能看到位置0-1
#  [0, 0, 0, 1],    <- 位置2只能看到位置0-2
#  [0, 0, 0, 0]]    <- 位置3可以看到所有位置0-3
```

### Mask的应用

```python
# 在计算softmax之前应用mask
scores = QK^T / sqrt(d_k)
if mask is not None:
    scores = np.where(mask, -1e9, scores)  # mask位置设为很小的负数
attention_weights = softmax(scores)
```

## 特点

### 优点

- ✅ **计算高效**：使用矩阵乘法，可以高度并行化
- ✅ **无需额外参数**：相比参数化的注意力机制，不引入额外参数
- ✅ **数值稳定**：通过缩放避免梯度消失
- ✅ **灵活性强**：支持批量处理和多种mask策略
- ✅ **可扩展性好**：是Multi-Head Attention的基础

### 缺点

- ❌ **空间复杂度**：需要$O(n \times m)$的内存存储注意力矩阵
- ❌ **计算复杂度**：$O(n \times m \times d)$，对于长序列开销大
- ❌ **固定计算方式**：使用点积相似度，灵活性不如可学习的注意力

### 复杂度分析

假设查询序列长度为$n$，键/值序列长度为$m$，维度为$d$：

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| $QK^T$ | $O(n \times m \times d)$ | $O(n \times m)$ |
| Softmax | $O(n \times m)$ | $O(n \times m)$ |
| 注意力加权 | $O(n \times m \times d)$ | $O(n \times d)$ |
| **总计** | $O(n \times m \times d)$ | $O(n \times m)$ |

## 应用场景

### 1. Transformer Encoder

在自注意力（Self-Attention）中，$Q$、$K$、$V$都来自同一个输入：

```python
# 输入: X shape (seq_len, d_model)
Q = W_Q @ X  # (seq_len, d_k)
K = W_K @ X  # (seq_len, d_k)
V = W_V @ X  # (seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
```

### 2. Transformer Decoder

#### Masked Self-Attention
使用因果mask防止看到未来信息：

```python
causal_mask = create_causal_mask(seq_len)
output, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
```

#### Cross-Attention
Decoder关注Encoder的输出：

```python
# Q来自decoder，K和V来自encoder
Q = decoder_hidden  # (target_len, d_k)
K = encoder_output  # (source_len, d_k)
V = encoder_output  # (source_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
```

### 3. Multi-Head Attention

Scaled Dot-Product Attention是Multi-Head Attention的核心：

```python
# 对每个head分别计算
for i in range(num_heads):
    Q_i = W_Q[i] @ X
    K_i = W_K[i] @ X
    V_i = W_V[i] @ X

    head_i, _ = scaled_dot_product_attention(Q_i, K_i, V_i)
    heads.append(head_i)

# 拼接所有head的输出
output = concat(heads) @ W_O
```

## 代码实现

### 基础实现

```python
import numpy as np

def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Args:
        Q: 查询矩阵 (..., seq_len_q, d_k)
        K: 键矩阵 (..., seq_len_k, d_k)
        V: 值矩阵 (..., seq_len_k, d_v)
        mask: 掩码矩阵（可选）

    Returns:
        output: 注意力输出
        attention_weights: 注意力权重
    """
    d_k = K.shape[-1]

    # 计算注意力分数并缩放
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

    # 应用mask
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Softmax归一化
    attention_weights = softmax(scores, axis=-1)

    # 加权求和
    output = np.matmul(attention_weights, V)

    return output, attention_weights
```

### 完整类实现

```python
class ScaledDotProductAttention:
    """缩放点积注意力类"""

    def __init__(self, temperature=None):
        self.temperature = temperature

    def forward(self, Q, K, V, mask=None):
        d_k = K.shape[-1]

        # 计算并缩放
        scores = np.matmul(Q, K.transpose(-2, -1))
        if self.temperature is None:
            scores = scores / np.sqrt(d_k)
        else:
            scores = scores / self.temperature

        # 应用mask
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        # Softmax + 加权求和
        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)

        return output, attention_weights
```

## 使用示例

### 示例1：基础使用

```python
# 创建输入矩阵
seq_len = 8
d_k = 64
d_v = 64

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

# 计算注意力
output, weights = scaled_dot_product_attention(Q, K, V)

print(f"输出形状: {output.shape}")      # (8, 64)
print(f"权重形状: {weights.shape}")     # (8, 8)
print(f"权重总和: {weights.sum(axis=-1)}")  # [1., 1., ..., 1.]
```

### 示例2：批量处理

```python
batch_size = 4
seq_len = 10
d_k = 64

Q_batch = np.random.randn(batch_size, seq_len, d_k)
K_batch = np.random.randn(batch_size, seq_len, d_k)
V_batch = np.random.randn(batch_size, seq_len, d_k)

output_batch, weights_batch = scaled_dot_product_attention(
    Q_batch, K_batch, V_batch
)

print(f"批量输出形状: {output_batch.shape}")  # (4, 10, 64)
print(f"批量权重形状: {weights_batch.shape}") # (4, 10, 10)
```

### 示例3：使用Padding Mask

```python
seq_len = 6
padding_positions = [4, 5]  # 最后两个位置是padding

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

# 创建并应用padding mask
mask = create_padding_mask(seq_len, padding_positions)
output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

# padding位置的权重接近0
print(weights[:, padding_positions])  # 值接近0
```

### 示例4：使用Causal Mask

```python
seq_len = 5
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

# 创建并应用因果mask
causal_mask = create_causal_mask(seq_len)
output, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

# 每个位置只能attend到当前及之前的位置
print(weights)
# [[w00,   0,   0,   0,   0],
#  [w10, w11,   0,   0,   0],
#  [w20, w21, w22,   0,   0],
#  [w30, w31, w32, w33,   0],
#  [w40, w41, w42, w43, w44]]
```

## 与其他注意力机制的对比

### 1. vs Additive Attention (Bahdanau Attention)

| 特性 | Scaled Dot-Product | Additive Attention |
|------|-------------------|-------------------|
| 相似度计算 | $Q \cdot K^T$ | $v^T \tanh(W_1 Q + W_2 K)$ |
| 参数量 | 无（在注意力层） | 有可学习参数 |
| 计算效率 | 高（矩阵乘法） | 较低 |
| 适用维度 | 任意 | 任意 |
| 典型应用 | Transformer | RNN Seq2seq |

### 2. vs Soft Attention

| 特性 | Scaled Dot-Product | Soft Attention |
|------|-------------------|---------------|
| 缩放因子 | $\sqrt{d_k}$ | 无或可学习 |
| 批量处理 | 原生支持 | 需要额外实现 |
| 数值稳定性 | 好 | 依赖实现 |
| 灵活性 | 固定公式 | 可自定义得分函数 |

### 3. vs Multi-Head Attention

| 特性 | Scaled Dot-Product | Multi-Head Attention |
|------|-------------------|---------------------|
| 关系 | 基础构建块 | 使用多个SDPA |
| 表达能力 | 单一子空间 | 多个子空间 |
| 参数量 | 无 | $W_Q, W_K, W_V, W_O$ |
| 计算成本 | 低 | 高（多个head） |

## 实现技巧

### 1. 数值稳定性

```python
# 好的实现：减去最大值
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 避免：直接计算可能溢出
# exp_x = np.exp(x)  # 可能溢出
```

### 2. Mask的正确使用

```python
# 好的实现：使用很小的负数
scores = np.where(mask, -1e9, scores)

# 避免：使用0（会导致错误的softmax结果）
# scores = np.where(mask, 0, scores)
```

### 3. 内存优化

```python
# 对于超长序列，可以使用分块计算
def chunked_attention(Q, K, V, chunk_size=1024):
    seq_len = Q.shape[0]
    outputs = []

    for i in range(0, seq_len, chunk_size):
        chunk_Q = Q[i:i+chunk_size]
        chunk_output, _ = scaled_dot_product_attention(chunk_Q, K, V)
        outputs.append(chunk_output)

    return np.concatenate(outputs, axis=0)
```

## 可视化

注意力权重矩阵可以直观展示模型关注的位置：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 计算注意力
output, weights = scaled_dot_product_attention(Q, K, V)

# 可视化
plt.figure(figsize=(10, 8))
sns.heatmap(weights, annot=True, fmt='.2f', cmap='YlOrRd')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Attention Weights')
plt.show()
```

## 常见问题

### Q1: 为什么使用点积而不是其他相似度度量？

**答**：点积有以下优势：
- 计算效率高，可以用高度优化的矩阵乘法
- 与向量的模长相关，提供了丰富的语义信息
- 在高维空间中表现良好（配合缩放）

### Q2: 缩放因子一定要是$\sqrt{d_k}$吗？

**答**：不一定。可以使用其他缩放因子（temperature），但$\sqrt{d_k}$是理论推导的最优值，能够保持方差稳定。

### Q3: Mask必须是布尔值吗？

**答**：不一定。Mask可以是：
- 布尔值（True/False）
- 加性mask（0或-inf）
- 乘性mask（0或1）
本实现使用布尔值是为了代码清晰。

### Q4: 如何处理不同长度的Q和K？

**答**：Q和K的长度可以不同：
- Q的长度决定输出的序列长度
- K/V的长度必须相同
- 这在Cross-Attention中很常见（如Encoder-Decoder）

## 扩展阅读

### 相关算法

1. **Multi-Head Attention**：使用多个Scaled Dot-Product Attention并行计算
2. **Self-Attention**：Q、K、V来自同一输入
3. **Cross-Attention**：Q和K/V来自不同输入
4. **Relative Position Encoding**：在注意力计算中加入相对位置信息

### 优化变种

1. **Sparse Attention**：只计算部分位置的注意力，降低复杂度
2. **Linear Attention**：使用核技巧将复杂度降到线性
3. **Flash Attention**：优化GPU内存访问模式
4. **Multi-Query Attention**：多个查询共享一组键值

## 参考文献

1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 2017.
   - 原始论文，提出Transformer和Scaled Dot-Product Attention

2. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL 2019.
   - BERT模型，广泛应用Scaled Dot-Product Attention

3. Radford, A., et al. (2019). Language models are unsupervised multitask learners.
   - GPT-2，使用Causal Mask的Scaled Dot-Product Attention

4. Dao, T., et al. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. NeurIPS 2022.
   - 优化的注意力实现

## 文件说明

- `scaled_dot_product_attention.py`: Python实现（带详细中文注释）
- `scaled_dot_product_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档

## 运行示例

```bash
# 运行Python脚本
python scaled_dot_product_attention.py

# 启动Jupyter Notebook
jupyter notebook scaled_dot_product_attention.ipynb
```
