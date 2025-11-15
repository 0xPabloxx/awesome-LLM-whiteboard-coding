# Transformer Decoder

Transformer Decoder是自回归生成模型的核心架构，包含Masked Self-Attention、Cross-Attention（可选）和Feed-Forward Network等组件。

## 📖 核心概念

### Decoder的组成

一个完整的Decoder Layer包含：

```
输入
  ↓
Masked Self-Attention  ← 只能看到当前及之前的token
  ↓
Add & LayerNorm
  ↓
Cross-Attention (可选)  ← 关注Encoder输出
  ↓
Add & LayerNorm
  ↓
Feed-Forward Network
  ↓
Add & LayerNorm
  ↓
输出
```

### 两种架构

#### 1. Decoder-only (GPT风格)

```python
class DecoderLayer:
    components = [
        "Masked Self-Attention",  # 因果注意力
        "Feed-Forward Network"     # 位置独立FFN
    ]
```

**应用：** GPT系列、LLaMA、PaLM、Claude等所有主流LLM

#### 2. Encoder-Decoder (T5风格)

```python
class DecoderLayer:
    components = [
        "Masked Self-Attention",   # Decoder内部
        "Cross-Attention",          # 关注Encoder
        "Feed-Forward Network"
    ]
```

**应用：** T5、BART、mT5等Seq2Seq模型

## 🎯 核心组件详解

### 1. Masked Self-Attention（因果注意力）

确保每个位置只能看到自己和之前的位置：

```python
# Causal Mask（下三角矩阵）
mask = np.tril(np.ones((seq_len, seq_len)))

# 例如 seq_len=5:
[[1, 0, 0, 0, 0],   # token 0 只能看到自己
 [1, 1, 0, 0, 0],   # token 1 可以看到 0,1
 [1, 1, 1, 0, 0],   # token 2 可以看到 0,1,2
 [1, 1, 1, 1, 0],   # ...
 [1, 1, 1, 1, 1]]   # token 4 可以看到所有
```

**作用：**
- 保证自回归生成的因果性
- 训练时可并行计算（所有位置同时）
- 推理时逐个生成token

**实现：**

```python
class MultiHeadSelfAttention:
    def forward(self, x, mask=None):
        # 计算attention scores
        scores = Q @ K.T / sqrt(d_k)

        # 应用causal mask
        if mask is not None:
            scores = where(mask == 0, -inf, scores)

        # Softmax + 加权求和
        attn_weights = softmax(scores)
        output = attn_weights @ V

        return output
```

### 2. Cross-Attention（交叉注意力）

Decoder关注Encoder的输出：

```python
# Query来自Decoder
Q = decoder_input @ W_q

# Key和Value来自Encoder
K = encoder_output @ W_k
V = encoder_output @ W_v

# 计算attention
scores = Q @ K.T / sqrt(d_k)
attn_weights = softmax(scores)
output = attn_weights @ V
```

**特点：**
- Q来自Decoder当前层
- K、V来自Encoder的最终输出
- 无需causal mask（可以看到全部Encoder输出）

**应用场景：**
- 机器翻译：Decoder关注源语言
- 摘要生成：Decoder关注原文
- 图像描述：Decoder关注图像特征

### 3. Feed-Forward Network

位置独立的全连接网络：

```python
class FFN:
    def forward(self, x):
        # 第一层：扩展维度
        hidden = GELU(x @ W1 + b1)  # (d → 4d)

        # 第二层：恢复维度
        output = hidden @ W2 + b2   # (4d → d)

        return output
```

**特点：**
- 对每个位置独立应用
- 中间层通常是4倍embed_dim
- 使用GELU激活函数（GPT）或ReLU（原始Transformer）

**参数量：**
```
W1: embed_dim × 4*embed_dim
W2: 4*embed_dim × embed_dim
总计: 8 × embed_dim²
```

### 4. Layer Normalization

稳定训练的关键：

```python
# Post-LN (BERT风格)
x = x + SubLayer(x)
x = LayerNorm(x)

# Pre-LN (GPT风格，更常用)
x = x + SubLayer(LayerNorm(x))
```

**Pre-LN优势：**
- 训练更稳定
- 不需要learning rate warmup
- 梯度流动更好

### 5. Residual Connection

改善梯度流动：

```python
# 每个子层都有residual
x = x + self_attention(x)
x = x + cross_attention(x)
x = x + ffn(x)
```

## 📊 参数量分析

### 单层Decoder参数

对于embed_dim = d：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Self-Attention (Q,K,V,O) | 4d² | ~33% |
| Cross-Attention (可选) | 4d² | ~33% |
| FFN (两层) | 8d² | ~67% |
| LayerNorm | 4d-6d | <1% |

**Decoder-only (无Cross-Attn)：** 约12d² 参数/层
**Encoder-Decoder (有Cross-Attn)：** 约16d² 参数/层

### 实际模型参数量

| 模型 | 层数 | embed_dim | 头数 | 总参数 |
|------|------|----------|------|--------|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Medium | 24 | 1024 | 16 | 345M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |
| LLaMA-7B | 32 | 4096 | 32 | 7B |
| LLaMA-13B | 40 | 5120 | 40 | 13B |
| LLaMA-70B | 80 | 8192 | 64 | 70B |

### 参数量公式

```python
def estimate_params(num_layers, embed_dim, vocab_size, has_cross_attn=False):
    # 每层参数
    if has_cross_attn:
        params_per_layer = 16 * embed_dim**2  # Self + Cross + FFN
    else:
        params_per_layer = 12 * embed_dim**2  # Self + FFN

    # 总参数
    total = num_layers * params_per_layer

    # Token Embedding + LM Head
    total += 2 * vocab_size * embed_dim

    # Position Embedding（如果使用）
    # total += max_seq_len * embed_dim

    return total
```

## 🏗️ 实现示例

### 完整Decoder Layer

```python
class TransformerDecoderLayer:
    def __init__(self, embed_dim, num_heads, has_cross_attention=False):
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)

        # 2. Cross-Attention (可选)
        if has_cross_attention:
            self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)
            self.ln2 = LayerNorm(embed_dim)

        # 3. FFN
        self.ffn = FeedForwardNetwork(embed_dim)
        self.ln3 = LayerNorm(embed_dim)

    def forward(self, x, encoder_output=None, causal_mask=None):
        # Self-Attention + Residual + LN
        x = self.ln1(x + self.self_attn(x, mask=causal_mask))

        # Cross-Attention + Residual + LN
        if hasattr(self, 'cross_attn') and encoder_output is not None:
            x = self.ln2(x + self.cross_attn(x, encoder_output))

        # FFN + Residual + LN
        x = self.ln3(x + self.ffn(x))

        return x
```

### 完整Decoder

```python
class TransformerDecoder:
    def __init__(self, num_layers, embed_dim, num_heads,
                 has_cross_attention=False, vocab_size=None):
        # 堆叠多层
        self.layers = [
            TransformerDecoderLayer(embed_dim, num_heads, has_cross_attention)
            for _ in range(num_layers)
        ]

        # 最终LayerNorm
        self.final_ln = LayerNorm(embed_dim)

        # Language Model Head
        if vocab_size is not None:
            self.lm_head = Linear(embed_dim, vocab_size)

    def forward(self, x, encoder_output=None, causal_mask=None):
        # 逐层处理
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask)

        # 最终LN
        x = self.final_ln(x)

        # 输出logits
        if hasattr(self, 'lm_head'):
            logits = self.lm_head(x)  # (seq_len, vocab_size)
            return logits

        return x
```

## 🎨 架构对比

### Decoder-only vs Encoder-Decoder

| 特性 | Decoder-only (GPT) | Encoder-Decoder (T5) |
|------|-------------------|---------------------|
| **组件** | Self-Attn + FFN | Self-Attn + Cross-Attn + FFN |
| **参数** | 12d² per layer | 16d² per layer |
| **输入** | 单序列 | 两个序列（src + tgt） |
| **训练** | 简单 | 复杂 |
| **推理** | 自回归生成 | 自回归生成 |
| **应用** | 通用LLM | Seq2Seq任务 |

### 为什么Decoder-only成为主流？

1. **架构简单**：只需一个模型
2. **训练简单**：单向语言建模
3. **效果好**：In-context learning能力强
4. **可扩展**：容易扩展到超大规模
5. **通用性**：一个模型完成多种任务

**现状：**
- GPT-3/4、LLaMA、PaLM、Claude等都是Decoder-only
- Encoder-Decoder主要用于特定的Seq2Seq任务

## 🚀 实际应用

### GPT系列

```python
# GPT-3配置
config = {
    'num_layers': 96,
    'embed_dim': 12288,
    'num_heads': 96,
    'vocab_size': 50257,
    'max_seq_len': 2048,
    'architecture': 'decoder-only'
}

参数量: 175B
应用: 文本生成、对话、代码生成等
```

### LLaMA

```python
# LLaMA-7B配置
config = {
    'num_layers': 32,
    'embed_dim': 4096,
    'num_heads': 32,
    'vocab_size': 32000,
    'max_seq_len': 4096,
    'architecture': 'decoder-only',
    'normalization': 'RMSNorm',  # 优化
    'attention': 'GQA'            # 显存优化
}

参数量: 7B
特点: 开源、高效
```

### T5

```python
# T5-Large配置
config = {
    'encoder_layers': 24,
    'decoder_layers': 24,
    'embed_dim': 1024,
    'num_heads': 16,
    'vocab_size': 32128,
    'architecture': 'encoder-decoder'
}

参数量: 770M
应用: 翻译、摘要、问答
```

## 💡 优化技巧

### 1. 训练优化

```python
# Pre-LN（更稳定）
x = x + SubLayer(LayerNorm(x))

# 梯度裁剪
gradients = clip_gradients(gradients, max_norm=1.0)

# 学习率调度
lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
```

### 2. 推理优化

```python
# KV Cache（必备）
kv_cache = KVCache(num_heads, head_dim, max_seq_len)

# Flash Attention（加速）
output = flash_attention(Q, K, V, causal=True)

# 量化（减少显存）
model = quantize(model, bits=8)  # INT8
```

### 3. 显存优化

```python
# Grouped-Query Attention
num_kv_heads = num_heads // 4  # GQA-4

# Activation Checkpointing
use_gradient_checkpointing = True

# Mixed Precision Training
use_fp16 = True  # 或 bfloat16
```

## 📈 性能基准

### 计算复杂度

对于序列长度n，嵌入维度d：

| 组件 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Self-Attention | O(n²d) | O(n²) |
| Cross-Attention | O(n·m·d) | O(n·m) |
| FFN | O(nd²) | O(d) |
| **总计** | **O(n²d + nd²)** | **O(n²)** |

### 显存占用（推理）

以LLaMA-7B为例（batch=1, seq_len=2048）：

```
模型权重: 7B × 2字节 (FP16) = 14GB
KV Cache: 32层 × 2 × 32头 × 2048 × 128 × 2字节 = 1GB
激活值: ~2GB

总计: 约17GB
```

## ⚠️ 常见问题

### 1. Causal Mask错误

```python
# 错误：忘记mask
output = self_attention(x)  # ❌ 会看到未来token

# 正确：应用causal mask
mask = create_causal_mask(seq_len)
output = self_attention(x, mask=mask)  # ✅
```

### 2. Cross-Attention位置

```python
# 错误顺序
x = cross_attention(x)  # ❌
x = self_attention(x)

# 正确顺序
x = self_attention(x)   # ✅ 先Self
x = cross_attention(x)  # 再Cross
```

### 3. LayerNorm位置

```python
# Post-LN（BERT，不太稳定）
x = LayerNorm(x + SubLayer(x))

# Pre-LN（GPT，更稳定）✅
x = x + SubLayer(LayerNorm(x))
```

## 🔗 相关资源

### 论文
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - 原始Transformer
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

### 开源实现
- **Transformers (HuggingFace)**: 最流行的实现
- **nanoGPT**: 极简GPT实现（教学用）
- **LLaMA**: Meta的开源LLM

## 🚀 快速开始

```bash
# 运行Python脚本
python decoder.py

# 运行Jupyter notebook
jupyter notebook decoder.ipynb
```

### 简单示例

```python
import numpy as np

# 创建GPT风格的Decoder
decoder = TransformerDecoder(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    has_cross_attention=False,
    vocab_size=50000
)

# 输入
x = np.random.randn(10, 768)  # (seq_len, embed_dim)
causal_mask = create_causal_mask(10)

# 前向传播
logits = decoder.forward(x, causal_mask=causal_mask, return_logits=True)
print(f"输出logits: {logits.shape}")  # (10, 50000)
```

---

**关键要点**：Transformer Decoder是所有自回归语言模型的核心，通过Masked Self-Attention保证因果性，结合FFN和残差连接，实现强大的序列生成能力。Decoder-only架构（GPT风格）已成为现代LLM的标准。
