# Multi-Head Attention (多头注意力)

## 概述

多头注意力（Multi-Head Attention）是Transformer架构的核心组件，由Vaswani等人在2017年的论文"Attention is All You Need"中提出。它通过使用多个注意力头并行计算，让模型能够同时关注输入序列在不同表示子空间的信息，极大提升了模型的表达能力。

多头注意力是所有现代大语言模型（BERT、GPT、T5等）的基础组件。

## 核心思想

多头注意力的关键特点：

1. **并行计算多个注意力头**：每个头独立关注不同的表示子空间
2. **捕获多样化的模式**：不同的头可以学习不同的注意力模式（句法、语义、位置等）
3. **参数高效**：通过减小每个头的维度，保持总参数量不变
4. **增强表达能力**：多个头的组合提供更丰富的表示

## 数学公式

给定输入序列 $X \in \mathbb{R}^{n \times d}$，多头注意力的计算如下：

$$
\begin{aligned}
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

其中：
- $h$ 是注意力头的数量
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}$ 是第$i$个头的投影矩阵
- $d_k = d / h$ 是每个头的维度
- $W^O \in \mathbb{R}^{d \times d}$ 是输出投影矩阵
- $\text{Attention}$ 是Scaled Dot-Product Attention

### Scaled Dot-Product Attention

每个头内部使用的注意力机制：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 详细步骤

### 1. 线性投影
对输入进行三次线性变换，得到Q、K、V：
```python
Q = X @ W_Q  # (seq_len, embed_dim)
K = X @ W_K  # (seq_len, embed_dim)
V = X @ W_V  # (seq_len, embed_dim)
```

### 2. 分割成多个头
将Q、K、V分割成h个头：
```python
# (seq_len, embed_dim) -> (num_heads, seq_len, head_dim)
Q_heads = split_heads(Q)
K_heads = split_heads(K)
V_heads = split_heads(V)
```

### 3. 并行计算注意力
对每个头独立计算Scaled Dot-Product Attention：
```python
for i in range(num_heads):
    head_i = Attention(Q_heads[i], K_heads[i], V_heads[i])
```

### 4. 合并头
将所有头的输出concatenate：
```python
# (num_heads, seq_len, head_dim) -> (seq_len, embed_dim)
concat_output = concat(head_1, ..., head_h)
```

### 5. 输出投影
通过输出投影矩阵$W^O$：
```python
output = concat_output @ W_O  # (seq_len, embed_dim)
```

## 架构图示

```
输入 X (seq_len, embed_dim)
         |
    ┌────┴────┬────────┬────────┐
    |         |        |        |
  W^Q       W^K      W^V      (线性投影)
    |         |        |
    Q         K        V
    |         |        |
    └────┬────┴────┬───┘
         |         |
     分割成 h 个头
         |
    ┌────┼────┬────┼────┐
 head_1  head_2 ... head_h  (并行计算attention)
    └────┼────┴────┼────┘
         |
      Concat
         |
       W^O          (输出投影)
         |
    输出 (seq_len, embed_dim)
```

## 代码实现

### 基础实现

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def split_heads(self, x):
        # (seq_len, embed_dim) -> (num_heads, seq_len, head_dim)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 0, 2)

    def forward(self, query, key, value, mask=None):
        # 1. 线性投影
        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v

        # 2. 分割成多头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 对每个头计算attention
        head_outputs = []
        for i in range(self.num_heads):
            head_out, _ = scaled_dot_product_attention(
                Q[i], K[i], V[i], mask
            )
            head_outputs.append(head_out)

        # 4. 合并头
        concat = self.combine_heads(np.stack(head_outputs))

        # 5. 输出投影
        output = concat @ self.W_o

        return output
```

## 多头注意力的优势

### 1. 捕获多样化的模式

不同的头可以学习关注不同的模式：
- **Head 1**：可能关注局部依赖（相邻词）
- **Head 2**：可能关注句法结构（主谓关系）
- **Head 3**：可能关注语义相关性
- **Head 4**：可能关注位置信息

### 2. 参数高效

假设嵌入维度为$d$，头数为$h$：

| 配置 | Q、K、V投影 | 每个头维度 | 总参数量 |
|------|------------|-----------|---------|
| 单头 | $d \times d$ × 3 | $d$ | $4d^2$ |
| 多头 | $d \times d$ × 3 | $d/h$ | $4d^2$ |

**参数量相同，但表达能力更强！**

### 3. 并行计算

所有头可以完全并行计算，提高效率。

## 应用场景

### 1. 自注意力（Self-Attention）
Q、K、V都来自同一输入：
```python
# Transformer Encoder/Decoder
output = multi_head_attention(x, x, x)
```

### 2. 跨注意力（Cross-Attention）
Q来自decoder，K、V来自encoder：
```python
# Transformer Decoder
output = multi_head_attention(
    query=decoder_input,
    key=encoder_output,
    value=encoder_output
)
```

### 3. 带Mask的注意力

#### 因果Mask（GPT）
```python
# 只能看到当前和之前的位置
causal_mask = np.tril(np.ones((seq_len, seq_len)))
output = multi_head_attention(x, x, x, mask=causal_mask)
```

#### Padding Mask
屏蔽padding位置，避免关注无效token。

## 复杂度分析

假设序列长度为$n$，嵌入维度为$d$，头数为$h$：

| 操作 | 复杂度 |
|------|--------|
| Q、K、V投影 | $O(n \cdot d^2)$ |
| 分割头 | $O(n \cdot d)$ |
| 每个头的attention | $O(h \cdot n^2 \cdot \frac{d}{h}) = O(n^2 \cdot d)$ |
| 合并头 | $O(n \cdot d)$ |
| 输出投影 | $O(n \cdot d^2)$ |
| **总计** | **$O(n^2 \cdot d + n \cdot d^2)$** |

当$n < d$时，主导项是$O(n \cdot d^2)$（投影）
当$n > d$时，主导项是$O(n^2 \cdot d)$（注意力计算）

## 参数配置

常见的Transformer模型配置：

| 模型 | embed_dim | num_heads | head_dim |
|------|-----------|-----------|----------|
| BERT-Base | 768 | 12 | 64 |
| BERT-Large | 1024 | 16 | 64 |
| GPT-2 Small | 768 | 12 | 64 |
| GPT-2 Large | 1280 | 20 | 64 |
| GPT-3 | 12288 | 96 | 128 |

**有趣的发现**：大多数模型的head_dim都在64-128之间！

## 多头 vs 单头对比

| 特性 | 单头注意力 | 多头注意力 |
|------|-----------|-----------|
| 表达能力 | 受限 | 强 |
| 关注模式 | 单一 | 多样化 |
| 参数量 | $4d^2$ | $4d^2$ (相同) |
| 计算效率 | 高 | 稍低（需要分割/合并） |
| 鲁棒性 | 低 | 高 |

## 可视化理解

### 注意力权重示例

假设8个头，每个头关注不同的模式：

```
句子: "The cat sat on the mat"

Head 1 (局部):
The  [0.8, 0.2, 0.0, 0.0, 0.0, 0.0]  # 关注相邻词
cat  [0.2, 0.6, 0.2, 0.0, 0.0, 0.0]

Head 2 (句法):
The  [0.1, 0.8, 0.1, 0.0, 0.0, 0.0]  # The修饰cat
sat  [0.0, 0.4, 0.4, 0.2, 0.0, 0.0]  # sat的主语是cat

Head 3 (语义):
cat  [0.1, 0.2, 0.1, 0.0, 0.0, 0.6]  # cat和mat语义相关
```

## 使用示例

### 示例1：标准自注意力
```python
# 参数
embed_dim = 512
num_heads = 8
seq_len = 20

# 创建层
mha = MultiHeadSelfAttention(embed_dim, num_heads)

# 输入
x = np.random.randn(seq_len, embed_dim)

# 前向传播
output, attention_weights = mha.forward(x, return_attention=True)

print(f"输入: {x.shape}")                    # (20, 512)
print(f"输出: {output.shape}")               # (20, 512)
print(f"注意力权重: {attention_weights.shape}")  # (8, 20, 20)
```

### 示例2：带因果Mask
```python
# GPT风格的自回归模型
causal_mask = np.tril(np.ones((seq_len, seq_len)))
output = mha.forward(x, mask=causal_mask)
```

### 示例3：跨注意力
```python
# Encoder-Decoder
encoder_out = np.random.randn(15, 512)  # encoder输出
decoder_in = np.random.randn(10, 512)   # decoder输入

cross_attn = MultiHeadAttention(embed_dim, num_heads)
output = cross_attn.forward(
    query=decoder_in,
    key=encoder_out,
    value=encoder_out
)
print(f"Cross-attention输出: {output.shape}")  # (10, 512)
```

## 实现技巧

### 1. 合并投影矩阵
为了提高效率，可以将所有头的投影矩阵合并成一个大矩阵：
```python
# 方法1：分别存储（慢）
W_q_heads = [W_q_1, W_q_2, ..., W_q_h]

# 方法2：合并矩阵（快）
W_q = np.random.randn(embed_dim, embed_dim)  # 包含所有头
```

### 2. 维度重塑
使用reshape和transpose来高效分割和合并头：
```python
# 分割
x = x.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

# 合并
x = x.transpose(1, 0, 2).reshape(seq_len, embed_dim)
```

### 3. 注意力Dropout
在实际应用中，通常在注意力权重上应用dropout：
```python
attention_weights = softmax(scores)
attention_weights = dropout(attention_weights, p=0.1)
output = attention_weights @ V
```

## 常见问题

### Q1: 为什么embed_dim必须能被num_heads整除？
A: 因为每个头需要平均分配维度。如果embed_dim=512, num_heads=8，那么每个头的维度是512/8=64。

### Q2: 多头数量如何选择？
A: 常见选择是8或16。更多的头可以捕获更多模式，但也增加计算开销。实践中通常保持head_dim=64。

### Q3: 多头注意力和单头有什么区别？
A: 参数量相同，但多头可以并行学习多种注意力模式，表达能力更强。

### Q4: 输出投影W^O的作用是什么？
A: 将concatenate后的多头输出进行线性变换，让模型学习如何组合不同头的信息。

### Q5: 多头注意力能否用于变长序列？
A: 可以，使用padding mask来屏蔽padding位置即可。

## 改进变体

### 1. Multi-Query Attention (MQA)
所有头共享同一个K和V，只有Q不同：
- 减少KV缓存
- 提升推理速度
- 用于PaLM等模型

### 2. Grouped-Query Attention (GQA)
介于MHA和MQA之间，将头分组共享KV：
- 平衡性能和效率
- 用于Llama 2等模型

### 3. Flash Attention
优化内存访问模式：
- 减少HBM访问
- 大幅提升训练速度
- 支持更长序列

## 参考文献

1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers. NAACL 2019.
3. Radford, A., et al. (2019). Language models are unsupervised multitask learners. (GPT-2)
4. Brown, T., et al. (2020). Language models are few-shot learners. NeurIPS. (GPT-3)
5. Dao, T., et al. (2022). FlashAttention: Fast and memory-efficient exact attention. NeurIPS.

## 文件说明

- `multi_head_attention.py`: Python实现（带详细中文注释）
- `multi_head_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档

## 总结

多头注意力是Transformer的核心创新，通过并行计算多个注意力头，在保持参数量不变的情况下，显著增强了模型的表达能力。它是现代大语言模型（BERT、GPT、T5等）不可或缺的组件。

**关键要点**：
- ✅ 多个头并行关注不同子空间
- ✅ 参数量与单头相同，表达能力更强
- ✅ 每个头可以学习不同的注意力模式
- ✅ 是所有Transformer架构的基础
