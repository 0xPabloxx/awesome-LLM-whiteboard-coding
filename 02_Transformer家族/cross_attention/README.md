# Cross-Attention (交叉注意力)

## 概述

交叉注意力（Cross-Attention）是Transformer编码器-解码器架构中的关键组件，用于连接两个不同的序列。它允许解码器序列关注编码器序列，实现源序列和目标序列之间的信息交互。

在经典的机器翻译任务中，交叉注意力让目标语言的每个词都能关注源语言的所有词，从而实现准确的翻译对齐。

## 核心思想

交叉注意力的关键特点：

1. **Query来自解码器（目标序列）**
2. **Key和Value来自编码器（源序列）**
3. **实现非对称的序列间交互**：解码器主动"查询"编码器的信息
4. **长度可以不同**：源序列和目标序列的长度不需要相同

## 数学公式

给定编码器输出 $E \in \mathbb{R}^{n \times d}$ 和解码器隐状态 $D \in \mathbb{R}^{m \times d}$：

$$
\begin{aligned}
Q &= DW_Q \quad \text{(Query来自解码器)} \\
K &= EW_K \quad \text{(Key来自编码器)} \\
V &= EW_V \quad \text{(Value来自编码器)} \\
\text{CrossAttention}(Q,K,V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{aligned}
$$

其中：
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ 是可学习的投影矩阵
- $n$ 是源序列长度，$m$ 是目标序列长度（可以不同）
- 注意力矩阵形状为 $\mathbb{R}^{m \times n}$（通常不是方阵）
- 输出维度为 $\mathbb{R}^{m \times d}$（与解码器输入维度相同）

## 详细步骤

### 1. 线性投影
```python
# Q来自解码器
Q = decoder_hidden @ W_Q  # (tgt_len, d_k)

# K、V来自编码器
K = encoder_output @ W_K  # (src_len, d_k)
V = encoder_output @ W_V  # (src_len, d_v)
```

### 2. 计算注意力得分
```python
# 注意：得到的是 (tgt_len × src_len) 的矩阵
scores = Q @ K.T / sqrt(d_k)  # (tgt_len, src_len)
```

### 3. Softmax归一化
```python
# 对每个目标位置，在源序列维度上做softmax
attention_weights = softmax(scores, axis=-1)  # (tgt_len, src_len)
```

### 4. 加权求和
```python
# 用编码器信息更新解码器
output = attention_weights @ V  # (tgt_len, d_v)
```

## 与Self-Attention的对比

这是理解Cross-Attention最重要的部分：

| 特性 | Self-Attention | Cross-Attention |
|------|----------------|-----------------|
| **Query来源** | 输入序列本身 | 解码器（目标序列） |
| **Key来源** | 输入序列本身 | 编码器（源序列） |
| **Value来源** | 输入序列本身 | 编码器（源序列） |
| **注意力矩阵形状** | 方阵 (n × n) | 通常非方阵 (m × n) |
| **关注范围** | 序列内部的所有位置 | 另一个序列的所有位置 |
| **用途** | 捕获序列内部依赖 | 连接两个不同序列 |
| **应用场景** | BERT、GPT内部 | 机器翻译、图像描述 |

### 直观理解

**Self-Attention（自注意力）**：
```
句子: "我 爱 自然 语言 处理"
每个词关注这个句子本身的所有词
→ "我"可以关注"爱"、"自然"、"语言"、"处理"
```

**Cross-Attention（交叉注意力）**：
```
源句子（英文）: "I love NLP"
目标句子（中文）: "我 爱 自然 语言 处理"

中文的每个词关注英文句子的所有词
→ "爱"主要关注"love"
→ "自然语言处理"主要关注"NLP"
```

## Mask机制

### Padding Mask

在实际应用中，批处理的序列长度不同，需要padding到相同长度。Padding mask确保模型不会关注padding位置。

```python
def create_padding_mask(src_len, tgt_len, valid_src_len):
    """
    创建padding mask

    Args:
        src_len: 源序列总长度（包括padding）
        tgt_len: 目标序列长度
        valid_src_len: 源序列的实际有效长度

    Returns:
        mask: (tgt_len, src_len)，有效位置为1，padding为0
    """
    mask = np.zeros((tgt_len, src_len))
    mask[:, :valid_src_len] = 1
    return mask

# 示例：源序列长度为10，但只有前7个是有效的
mask = create_padding_mask(src_len=10, tgt_len=8, valid_src_len=7)
# 后3个位置的注意力权重会被设置为0
```

**注意**：Cross-Attention通常不需要因果mask（causal mask），因为它关注的是编码器输出，而不是解码器自身。因果mask主要用于解码器的Self-Attention层。

## 代码实现

### 基础实现

```python
class CrossAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        # Q用于投影解码器，K、V用于投影编码器
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, decoder_input, encoder_output, mask=None):
        # Q来自解码器
        Q = decoder_input @ self.W_q
        # K、V来自编码器
        K = encoder_output @ self.W_k
        V = encoder_output @ self.W_v

        # 计算注意力得分
        scores = Q @ K.T / np.sqrt(self.embed_dim)

        # 应用mask
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax + 加权求和
        weights = softmax(scores, axis=-1)
        output = weights @ V

        return output, weights
```

### 使用示例

```python
# 创建交叉注意力层
cross_attn = CrossAttention(embed_dim=512)

# 编码器输出（源序列）
encoder_output = np.random.randn(10, 512)  # (src_len=10, embed_dim=512)

# 解码器输入（目标序列）
decoder_input = np.random.randn(8, 512)   # (tgt_len=8, embed_dim=512)

# 前向传播
output, weights = cross_attn.forward(decoder_input, encoder_output)

print(f"输出: {output.shape}")       # (8, 512) - 与decoder长度相同
print(f"注意力权重: {weights.shape}")  # (8, 10) - tgt_len × src_len
```

## 在Transformer中的位置

在标准的Transformer Decoder中，每个解码器层包含三个子层：

```
解码器层结构：
┌─────────────────────────────────────┐
│ 1. Masked Self-Attention            │  ← 解码器自己看自己
│    (带因果mask)                      │
├─────────────────────────────────────┤
│ 2. Cross-Attention ⭐                │  ← 解码器看编码器（本文重点）
│    - Q来自上一层输出                  │
│    - K、V来自编码器输出               │
├─────────────────────────────────────┤
│ 3. Feed-Forward Network             │
└─────────────────────────────────────┘
```

数据流向：
1. **编码器输出** → 所有解码器层的Cross-Attention的K、V
2. **解码器输入** → 第一个解码器层
3. **每层解码器**：
   - Self-Attention（看自己）
   - Cross-Attention（看编码器）
   - FFN
4. **最后一层输出** → 线性层 + Softmax → 预测

## 应用场景

### 1. 机器翻译（最经典）

**任务**：英文 → 中文

```python
源句子: "I love machine learning"
目标句子: "我 爱 机器 学习"

# Cross-Attention让每个中文词关注英文句子
"我"   → 主要关注 "I"
"爱"   → 主要关注 "love"
"机器" → 主要关注 "machine"
"学习" → 主要关注 "learning"
```

### 2. 图像描述生成

**任务**：图像 → 文字描述

```python
编码器: CNN提取图像特征（不同区域）
解码器: 生成描述文本

# Cross-Attention让每个生成的词关注相关的图像区域
"一只" → 关注检测到动物的区域
"猫"   → 关注猫的特征区域
"在"   → 关注位置相关区域
"睡觉" → 关注猫的姿态区域
```

### 3. 语音识别

**任务**：音频 → 文本

```python
编码器: 音频帧的特征序列
解码器: 识别出的文本

# Cross-Attention让每个字符关注对应的音频帧
```

### 4. 问答系统

**任务**：(问题 + 文档) → 答案

```python
编码器: 处理问题和文档
解码器: 生成答案

# Cross-Attention让答案的每个词关注问题和文档的相关部分
```

## 特点分析

### 优点

- ✅ **灵活的长度**：源序列和目标序列长度可以不同
- ✅ **直接对齐**：显式建模序列间的对应关系
- ✅ **信息融合**：解码器可以访问编码器的全局信息
- ✅ **可解释性**：注意力权重可视化显示对齐关系
- ✅ **梯度流动**：提供从解码器到编码器的直接梯度路径

### 缺点

- ❌ **计算开销**：需要计算 O(m×n) 的注意力矩阵
- ❌ **内存消耗**：存储 m×n 的注意力权重
- ❌ **依赖编码器**：解码器必须等待编码器完成

## 复杂度分析

| 操作 | 复杂度 |
|------|--------|
| Q投影（解码器） | O(m·d²) |
| K、V投影（编码器） | O(n·d²) |
| 计算QK^T | O(m·n·d) |
| Softmax | O(m·n) |
| 加权求和 | O(m·n·d) |
| **总计** | **O((m+n)·d² + m·n·d)** |

其中：
- m = 目标序列长度
- n = 源序列长度
- d = 嵌入维度

当 m 和 n 都很大时，主导项是 O(m·n·d)。

## 注意力权重的解释

交叉注意力权重矩阵 $A \in \mathbb{R}^{m \times n}$：

- **行（row）**：目标序列的一个位置对源序列所有位置的注意力分布
- **列（column）**：源序列的一个位置被目标序列所有位置关注的程度
- **每行和为1**：softmax归一化保证

示例（机器翻译）：
```
       I    love  machine  learning
我    0.8   0.1    0.05     0.05
爱    0.1   0.8    0.05     0.05
机器  0.05  0.05   0.8      0.1
学习  0.05  0.05   0.1      0.8
```

解读：
- "我"最关注"I"（权重0.8）
- "爱"最关注"love"（权重0.8）
- "机器"最关注"machine"（权重0.8）

## 实现细节

### 1. 批处理实现

```python
def forward_batch(self, decoder_input, encoder_output, mask=None):
    """
    批处理版本

    Args:
        decoder_input: (batch_size, tgt_len, embed_dim)
        encoder_output: (batch_size, src_len, embed_dim)
        mask: (batch_size, tgt_len, src_len)

    Returns:
        output: (batch_size, tgt_len, embed_dim)
        attention_weights: (batch_size, tgt_len, src_len)
    """
    batch_size, tgt_len, embed_dim = decoder_input.shape
    src_len = encoder_output.shape[1]

    # 投影：(batch, len, dim) @ (dim, dim) → (batch, len, dim)
    Q = decoder_input @ self.W_q
    K = encoder_output @ self.W_k
    V = encoder_output @ self.W_v

    # 注意力得分：(batch, tgt_len, dim) @ (batch, dim, src_len)
    #           → (batch, tgt_len, src_len)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(embed_dim)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    weights = softmax(scores, axis=-1)
    output = weights @ V

    return output, weights
```

### 2. 多头Cross-Attention

```python
class MultiHeadCrossAttention:
    """多头交叉注意力"""

    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 每个头有自己的Q、K、V投影
        self.heads = [
            CrossAttention(self.head_dim)
            for _ in range(num_heads)
        ]

        # 输出投影
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, decoder_input, encoder_output, mask=None):
        # 分头
        head_outputs = []
        for head in self.heads:
            output, _ = head.forward(decoder_input, encoder_output, mask)
            head_outputs.append(output)

        # 拼接
        concat = np.concatenate(head_outputs, axis=-1)

        # 输出投影
        output = concat @ self.W_o

        return output
```

## PyTorch实现参考

```python
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, decoder_input, encoder_output, mask=None):
        # decoder_input: (batch, tgt_len, embed_dim)
        # encoder_output: (batch, src_len, embed_dim)

        Q = self.query(decoder_input)
        K = self.key(encoder_output)
        V = self.value(encoder_output)

        # 注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weights, V)

        return output, attn_weights
```

## 调试技巧

### 1. 检查形状
```python
print(f"Q shape: {Q.shape}")  # 应该是 (tgt_len, embed_dim)
print(f"K shape: {K.shape}")  # 应该是 (src_len, embed_dim)
print(f"V shape: {V.shape}")  # 应该是 (src_len, embed_dim)
print(f"scores shape: {scores.shape}")  # 应该是 (tgt_len, src_len)
print(f"output shape: {output.shape}")  # 应该是 (tgt_len, embed_dim)
```

### 2. 检查注意力权重
```python
# 每行应该和为1
row_sums = attention_weights.sum(axis=-1)
assert np.allclose(row_sums, 1.0), "注意力权重每行应该和为1"

# 所有值应该在[0, 1]之间
assert (attention_weights >= 0).all() and (attention_weights <= 1).all()
```

### 3. 可视化注意力
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, annot=True, fmt='.2f', cmap='YlOrRd')
plt.xlabel('Source Positions')
plt.ylabel('Target Positions')
plt.title('Cross-Attention Weights')
plt.show()
```

## 常见问题

### Q1: Cross-Attention何时使用？
**A**: 当你有两个不同的序列需要交互时。最常见的是Encoder-Decoder架构中，解码器需要访问编码器的信息。

### Q2: 为什么Cross-Attention不需要因果mask？
**A**: 因果mask用于防止"看到未来"，主要用于解码器的Self-Attention。而Cross-Attention关注的是编码器输出（已完全编码），不存在"未来"的概念。

### Q3: Cross-Attention的K和V可以不同吗？
**A**: 理论上可以，但在标准Transformer中，K和V都来自编码器输出。有些变体可能会使用不同的K和V。

### Q4: Cross-Attention能双向吗？
**A**: 标准的Cross-Attention是单向的（decoder→encoder）。但可以设计双向的，例如让编码器也关注解码器。

### Q5: 注意力权重可以用于对齐吗？
**A**: 是的！在机器翻译中，Cross-Attention权重常被可视化为词对齐矩阵，显示源语言和目标语言的对应关系。

## 扩展阅读

### 改进变体

1. **Multi-Query Attention**：K、V在多头之间共享，减少参数
2. **Grouped-Query Attention**：介于Multi-Head和Multi-Query之间
3. **Flash Attention**：优化内存访问模式，加速计算
4. **Local Cross-Attention**：只关注局部窗口，降低复杂度

### 应用拓展

1. **Vision Transformer**：图像patch之间的交叉注意力
2. **CLIP**：图像和文本之间的交叉注意力
3. **Flamingo**：视觉和语言的交叉注意力
4. **Perceiver**：任意模态到潜在表示的交叉注意力

## 参考文献

1. Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS.
   - 提出Cross-Attention作为Transformer Decoder的关键组件

2. Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015.
   - 早期的注意力机制，Cross-Attention的前身

3. Luong, M., et al. (2015). "Effective Approaches to Attention-based Neural Machine Translation." EMNLP.
   - 改进的注意力机制

4. Xu, K., et al. (2015). "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." ICML.
   - 图像描述中的交叉注意力应用

## 文件说明

- `cross_attention.py`: Python实现（带详细中文注释）
- `cross_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档

## 快速开始

```bash
# 运行Python脚本
python cross_attention.py

# 或打开Jupyter Notebook
jupyter notebook cross_attention.ipynb
```

## 总结

Cross-Attention是连接两个序列的桥梁，是Encoder-Decoder架构的核心：

- 🎯 **目的**：让目标序列能够访问源序列的信息
- 🔑 **关键**：Q来自一个序列，K、V来自另一个序列
- 📐 **形状**：注意力矩阵通常不是方阵（m × n）
- 🌟 **应用**：机器翻译、图像描述、语音识别等所有Seq2Seq任务
