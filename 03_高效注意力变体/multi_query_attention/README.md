# Multi-Query Attention (MQA，多查询注意力)

## 概述

Multi-Query Attention (MQA) 是一种极致高效的注意力机制，由Noam Shazeer在2019年的论文"Fast Transformer Decoding: One Write-Head is All You Need"中提出。

MQA让所有查询头共享同一对键值头，将KV缓存减少到最小，是目前推理效率最高的注意力机制。被应用于PaLM、Falcon、StarCoder等超大规模语言模型。

## 核心思想

MQA的关键特点：

1. **共享KV头**：所有查询头使用同一对键值头
2. **极致优化**：KV缓存减少num_heads倍（最大化）
3. **参数最少**：K,V投影矩阵只有1个头的维度
4. **速度最快**：推理时内存和计算开销最小

## 数学公式

给定输入序列 $X \in \mathbb{R}^{n \times d}$，MQA的计算如下：

$$
\begin{aligned}
\text{MQA}(Q,K,V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW^K, VW^V)
\end{aligned}
$$

**核心区别**：所有头共享同一个K和V！

其中：
- $h$ 是查询头的数量
- $W_i^Q \in \mathbb{R}^{d \times d_k}$ 是第$i$个查询头的投影矩阵（$h$个）
- $W^K, W^V \in \mathbb{R}^{d \times d_k}$ 是共享的键值投影矩阵（各1个）
- $d_k = d / h$ 是每个头的维度

### 与MHA/GQA的对比

| 机制 | Q头数 | K头数 | V头数 | 共享方式 |
|------|-------|-------|-------|---------|
| **MHA** | h | h | h | 不共享，每个Q头有专属KV头 |
| **GQA** | h | g (g<h) | g (g<h) | 分组共享，每组Q头共享一对KV头 |
| **MQA** | h | 1 | 1 | 完全共享，所有Q头共享唯一KV头 |

## 详细步骤

### 1. 线性投影
对输入进行投影，注意K和V只有1个头的维度：
```python
Q = X @ W_Q  # (seq_len, embed_dim) - 包含所有h个Q头
K = X @ W_K  # (seq_len, head_dim) - 只有1个K头！
V = X @ W_V  # (seq_len, head_dim) - 只有1个V头！
```

### 2. 分割查询头
```python
# 只分割Q成多个头
Q_heads = split_heads(Q)  # (num_heads, seq_len, head_dim)

# K和V保持单头，不分割
# K: (seq_len, head_dim)
# V: (seq_len, head_dim)
```

### 3. MQA的核心：共享KV头
```python
for i in range(num_heads):
    # 所有查询头都使用相同的K和V
    head_i = Attention(Q_heads[i], K, V)  # K和V对所有头相同！
```

### 4. 合并头并输出投影
```python
concat_output = concat(head_1, ..., head_h)
output = concat_output @ W_O
```

## 架构图示

```
输入 X (seq_len, embed_dim)
         |
    ┌────┴────┬────────┬────────┐
    |         |        |        |
  W^Q       W^K      W^V      (投影)
    |         |        |
    Q         K        V
 (h个头)   (1个头)  (1个头)   ← MQA关键！
    |         |        |
    |         └────┬───┘
    |              |
    └──────┬───────┘
           |
    所有Q头共享KV
           |
    ┌──┬──┼──┬──┬──┐
    Q0 Q1 Q2 ... Qh  → 都使用同一KV
    └──┴──┼──┴──┴──┘
           |
        Concat
           |
         W^O
           |
    输出 (seq_len, embed_dim)
```

**示意图：8个Q头，1个KV头**
```
┌─────────────────────────────────────┐
│  唯一的KV头（被所有Q头共享）         │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐ │
│  │Q0 │Q1 │Q2 │Q3 │Q4 │Q5 │Q6 │Q7 │ │
│  └───┴───┴───┴───┴───┴───┴───┴───┘ │
│           ↓                          │
│      共享的 K, V                     │
└─────────────────────────────────────┘
```

## 代码实现

### 基础实现

```python
class MultiQueryAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q投影：完整的num_heads个头
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # K,V投影：只有1个头（MQA的核心）
        self.W_k = np.random.randn(embed_dim, self.head_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, self.head_dim) / np.sqrt(embed_dim)

        # 输出投影
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, x):
        # 1. 投影
        Q = x @ self.W_q  # (seq_len, embed_dim)
        K = x @ self.W_k  # (seq_len, head_dim) - 只有1个头
        V = x @ self.W_v  # (seq_len, head_dim) - 只有1个头

        # 2. 分割Q成多个头
        Q = self.split_heads(Q)  # (num_heads, seq_len, head_dim)

        # 3. 所有Q头共享同一KV头
        head_outputs = []
        for i in range(self.num_heads):
            # 关键：所有头使用相同的K和V
            head_out = attention(Q[i], K, V)
            head_outputs.append(head_out)

        # 4. 合并并输出投影
        concat = self.combine_heads(np.stack(head_outputs))
        output = concat @ self.W_o

        return output
```

## 参数量和KV缓存分析

假设 `embed_dim=512`, `num_heads=8`, `seq_len=1024`：

### 参数量对比

| 机制 | Q投影 | K投影 | V投影 | O投影 | 总计 | 相对MHA |
|------|-------|-------|-------|-------|------|---------|
| **MHA** | 512×512 | 512×512 | 512×512 | 512×512 | 1,048,576 | 100% |
| **GQA (4 KV)** | 512×512 | 512×256 | 512×256 | 512×512 | 786,432 | 75% |
| **GQA (2 KV)** | 512×512 | 512×128 | 512×128 | 512×512 | 655,360 | 62.5% |
| **MQA (1 KV)** | 512×512 | 512×64 | 512×64 | 512×512 | 589,824 | 56.3% |

**MQA参数量减少**：$2d^2 \times (1 - \frac{1}{h})$

### KV缓存对比

在推理时，每生成一个token需要缓存其K和V：

| 机制 | KV头数 | KV缓存大小 (seq_len=1024) | 相对MHA | 节省 |
|------|--------|---------------------------|---------|------|
| **MHA** | 8 | 2 × 1024 × 8 × 64 = 1,048,576 | 100% | 0% |
| **GQA (4 KV)** | 4 | 2 × 1024 × 4 × 64 = 524,288 | 50% | 50% |
| **GQA (2 KV)** | 2 | 2 × 1024 × 2 × 64 = 262,144 | 25% | 75% |
| **MQA (1 KV)** | 1 | 2 × 1024 × 1 × 64 = 131,072 | 12.5% | **87.5%** |

**MQA的KV缓存公式**：
$$
\text{KV Cache} = 2 \times \text{seq\_len} \times 1 \times \text{head\_dim}
$$

相比MHA减少：$(h-1)/h \times 100\%$，当$h=8$时节省**87.5%**！

## MQA的优势

### 1. 最大化减少KV缓存

**为什么KV缓存重要？**

在自回归生成中：
- 每生成一个新token，需要保存其K和V向量
- 需要使用所有历史token的KV缓存
- KV缓存大小 = 2 × seq_len × num_kv_heads × head_dim

**MQA的优势**：
- num_kv_heads = 1（最小值）
- KV缓存是MHA的 1/num_heads
- 对于8头模型，节省87.5%的缓存

### 2. 推理速度最快

**速度提升来源**：
1. **内存访问更少**：KV缓存小，内存带宽压力小
2. **计算更快**：K,V投影矩阵更小
3. **批处理更大**：节省的内存可用于更大的batch

**实测效果**（PaLM论文）：
- 推理速度提升：~30-50%
- 可支持更大的batch size
- 长序列生成尤其明显

### 3. 参数量最少

**参数减少**：
- K投影：从 $d \times d$ 减少到 $d \times (d/h)$
- V投影：从 $d \times d$ 减少到 $d \times (d/h)$
- 总计减少：约44%（8头时）

**好处**：
- 模型更紧凑
- 加载更快
- 微调更容易

### 4. 长序列生成优势

**示例（seq_len=4096，8头，head_dim=128，FP32）**：

```python
# MHA的KV缓存
mha_cache = 2 × 4096 × 8 × 128 × 4bytes = 33.5 MB

# MQA的KV缓存
mqa_cache = 2 × 4096 × 1 × 128 × 4bytes = 4.2 MB

# 节省：29.3 MB (87.5%)
```

**在批量推理时（batch_size=32）**：
- MHA需要：1.07 GB
- MQA需要：134 MB
- 节省：**936 MB**

这使得可以：
- 在同样GPU上处理更多请求
- 支持更长的上下文
- 部署更大的模型

## MQA的劣势

### 1. 可能降低模型质量

**原因**：
- 所有Q头共享同一KV头
- 表达能力受限于单一的K和V
- 只能通过不同的Q投影产生多样性

**实验结果**：
- PaLM论文：质量损失很小（<1% perplexity）
- 某些任务上MQA不如MHA/GQA
- 通过增加训练可以部分弥补

### 2. 表达能力受限

**对比**：
- **MHA**：8个不同的KV头 → 8种不同的K和V表示
- **GQA**：2个不同的KV头 → 2种不同的K和V表示
- **MQA**：1个KV头 → 只有1种K和V表示

**影响**：
- 捕获的模式多样性较低
- 某些复杂依赖关系可能学习不好
- 需要更大的模型容量来弥补

### 3. 不适合所有场景

**不推荐使用MQA的情况**：
- 模型规模较小（<10B）→ 用MHA或GQA
- 对质量要求极高 → 用MHA
- 训练资源有限 → GQA是更好的折中

## 实际应用

### PaLM 540B

Google的PaLM模型全面采用MQA：

```python
# PaLM 540B配置
embed_dim = 18432
num_heads = 48
num_kv_heads = 1  # MQA
seq_len = 2048
```

**选择MQA的原因**：
- 540B超大规模，KV缓存是主要瓶颈
- 推理时内存占用必须最小化
- 实验表明质量损失可接受
- 显著提升serving吞吐量

### Falcon 40B

```python
# Falcon 40B配置
embed_dim = 8192
num_heads = 128
num_kv_heads = 1  # MQA
```

**Falcon的发现**：
- MQA使得40B模型可以在单GPU推理
- 生成速度比MHA快约40%
- 质量与使用GQA相当

### StarCoder

```python
# StarCoder 15B配置
embed_dim = 6144
num_heads = 48
num_kv_heads = 1  # MQA
```

**代码生成场景**：
- 需要生成很长的代码
- MQA减少长序列的KV缓存压力
- 推理速度提升明显

## 使用示例

### 示例1：标准MQA

```python
# 创建MQA层
mqa = MultiQueryAttention(
    embed_dim=512,
    num_heads=8
)

# 前向传播
x = np.random.randn(seq_len, 512)
output = mqa.forward(x)

print(f"输入: {x.shape}")    # (seq_len, 512)
print(f"输出: {output.shape}")  # (seq_len, 512)
```

### 示例2：带因果Mask（自回归）

```python
# 创建因果mask
causal_mask = np.tril(np.ones((seq_len, seq_len)))

# 应用mask
output = mqa.forward(x, mask=causal_mask)
```

### 示例3：比较不同机制

```python
# 创建三种注意力机制
from multi_query_attention import compare_attention_variants

compare_attention_variants(
    embed_dim=512,
    num_heads=8,
    seq_len=1024
)

# 输出：
# MHA: KV缓存 1,048,576
# GQA: KV缓存 262,144 (节省75%)
# MQA: KV缓存 131,072 (节省87.5%)
```

## MQA vs GQA vs MHA 决策树

```
是否需要最高质量？
├─ 是 → 使用 MHA
└─ 否 → 是否是超大模型（>100B）？
        ├─ 是 → 使用 MQA
        └─ 否 → 是否需要平衡质量和效率？
                ├─ 是 → 使用 GQA
                └─ 否 → 推理速度是否是首要考虑？
                        ├─ 是 → 使用 MQA
                        └─ 否 → 使用 GQA
```

## 最佳实践

### 1. 何时使用MQA

**推荐场景**：
- ✅ 超大规模模型（>100B参数）
- ✅ 长序列生成（>2048 tokens）
- ✅ 推理速度是首要目标
- ✅ 资源极度受限的部署环境
- ✅ 需要高吞吐量的serving

**不推荐场景**：
- ❌ 小模型（<10B）
- ❌ 质量要求极高的任务
- ❌ 训练资源充足时（考虑GQA）

### 2. 如何评估MQA

**关键指标**：
1. **质量损失**：与MHA对比perplexity/accuracy
2. **速度提升**：推理时间减少百分比
3. **内存节省**：KV缓存减少量
4. **吞吐量**：每秒处理的token数

### 3. 从MHA迁移到MQA

**方法1：平均合并**（快速但质量略低）
```python
# 将MHA的多个KV头平均合并成1个
K_mqa = np.mean(K_mha_heads, axis=0)
V_mqa = np.mean(V_mha_heads, axis=0)
```

**方法2：训练蒸馏**（慢但质量高）
```python
# 使用MHA作为教师模型蒸馏到MQA
loss = distillation_loss(mqa_output, mha_output)
```

**方法3：继续预训练**（最佳）
```python
# 加载MHA模型，替换为MQA，继续训练
# 只需少量步数即可恢复性能
```

## 复杂度分析

假设序列长度为$n$，嵌入维度为$d$，头数为$h$：

| 操作 | MHA | MQA |
|------|-----|-----|
| Q投影 | $O(n \cdot d^2)$ | $O(n \cdot d^2)$ |
| K,V投影 | $O(n \cdot d^2)$ | $O(n \cdot d \cdot \frac{d}{h})$ |
| 注意力计算 | $O(n^2 \cdot d)$ | $O(n^2 \cdot d)$ |
| 输出投影 | $O(n \cdot d^2)$ | $O(n \cdot d^2)$ |
| **总时间** | $O(n^2 \cdot d + n \cdot d^2)$ | $O(n^2 \cdot d + n \cdot d^2)$ |
| **总空间（训练）** | $O(n^2 + n \cdot d)$ | $O(n^2 + n \cdot d)$ |
| **KV缓存（推理）** | $O(n \cdot d)$ | $O(n \cdot \frac{d}{h})$ |

**关键观察**：
- 训练时时间复杂度相同
- **推理时KV缓存减少h倍**（这是最大优势）
- K,V投影的参数量减少$(h-1)/h$

## 可视化理解

### KV头共享示意

```
MHA (8个KV头):
Q0 → K0, V0
Q1 → K1, V1
Q2 → K2, V2
...
Q7 → K7, V7

GQA (2个KV头):
Q0, Q1, Q2, Q3 → K0, V0
Q4, Q5, Q6, Q7 → K1, V1

MQA (1个KV头):
Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7 → K, V
       ↓
  所有Q头共享同一KV
```

### 长序列生成中的内存占用

```
假设：seq_len=4096, embed_dim=4096, num_heads=32, FP32

生成位置    MHA KV缓存    MQA KV缓存    节省
─────────────────────────────────────────────
token 1     64 KB         2 KB         96.9%
token 512   32 MB         1 MB         96.9%
token 2048  128 MB        4 MB         96.9%
token 4096  256 MB        8 MB         96.9%
```

## 常见问题

### Q1: MQA真的会降低质量吗？
A: 在大多数情况下质量损失很小（<1%）。PaLM、Falcon等模型证明了MQA的可行性。对于超大模型，通过增加容量可以弥补。

### Q2: MQA适合什么样的模型规模？
A: 通常用于100B+的超大模型。对于小模型（<10B），MHA或GQA可能更合适。

### Q3: 能否从MHA模型转换为MQA？
A: 可以。通过平均合并KV头或继续训练，可以从MHA转换到MQA，只需少量微调。

### Q4: MQA和GQA如何选择？
A: GQA是更安全的选择（质量接近MHA），MQA是更激进的选择（效率最高）。如果质量损失可接受，选MQA；否则选GQA。

### Q5: MQA在训练时也有优势吗？
A: 训练时优势不大（时间复杂度相同），主要优势在推理时（KV缓存小）。但参数少可以加快checkpoint保存/加载。

### Q6: 为什么不是所有模型都用MQA？
A: 因为有质量权衡。对于小模型或质量敏感任务，MHA/GQA更合适。MQA适合大模型和效率优先场景。

## 实现技巧

### 1. 高效的KV投影

```python
# 方法1：直接投影到head_dim
K = x @ W_k  # (seq_len, head_dim)

# 方法2：先投影再取第一个头（兼容MHA代码）
K_all = x @ W_k_full
K = K_all[:, :head_dim]
```

### 2. 共享KV的广播

```python
# 如果需要广播KV到所有头（某些实现）
K_repeated = K.unsqueeze(0).repeat(num_heads, 1, 1)
# 但这会增加内存，不推荐
```

### 3. 统一接口（支持MHA/GQA/MQA）

```python
class UnifiedAttention:
    def __init__(self, embed_dim, num_heads, num_kv_heads=None):
        if num_kv_heads is None:
            num_kv_heads = num_heads  # MHA
        # num_kv_heads = 1 时自动变成MQA
        # num_kv_heads在1和num_heads之间时是GQA
```

## 参考文献

1. **MQA原论文**：Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.

2. **PaLM应用**：Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways." arXiv:2204.02311.

3. **Falcon**：Penedo, G., et al. (2023). "The RefinedWeb Dataset for Falcon LLM." arXiv:2306.01116.

4. **StarCoder**：Li, R., et al. (2023). "StarCoder: May the source be with you!" arXiv:2305.06161.

5. **GQA对比**：Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245.

6. **Multi-Head Attention**：Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS.

## 文件说明

- `multi_query_attention.py`: Python实现（带详细中文注释）
- `multi_query_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档

## 总结

Multi-Query Attention是注意力机制优化的极致方案，通过让所有查询头共享单一的键值头，实现了最大化的KV缓存减少和推理加速。

**核心要点**：
- ✅ 所有Q头共享1对KV头
- ✅ KV缓存减少num_heads倍（87.5%对于8头）
- ✅ 推理速度最快，内存占用最小
- ✅ 被PaLM、Falcon、StarCoder等超大模型采用
- ✅ 是部署超大模型的关键技术

**适用场景**：
- 超大规模模型（>100B参数）
- 长序列生成任务
- 推理效率是首要目标
- 资源极度受限的环境

**权衡考虑**：
- ⚠️ 可能轻微降低质量（通常<1%）
- ⚠️ 需要权衡效率和质量
- ⚠️ GQA是更保守的选择

MQA代表了"用更少资源做更多事"的工程哲学，是理解现代LLM推理优化不可或缺的技术！
