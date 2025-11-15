# Grouped-Query Attention (GQA，分组查询注意力)

## 概述

Grouped-Query Attention (GQA) 是一种介于Multi-Head Attention (MHA)和Multi-Query Attention (MQA)之间的高效注意力机制，由Ainslie等人在2023年的论文"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"中提出。

GQA通过让多个查询头共享键值头，在保持模型质量的同时显著减少了KV缓存，是目前大语言模型（如Llama 2、Mistral）中广泛采用的高效注意力方案。

## 核心思想

GQA的关键特点：

1. **分组共享**：将查询头分成G组，每组共享一对键值头
2. **平衡设计**：在MHA的质量和MQA的效率之间取得平衡
3. **减少缓存**：KV缓存减少G倍，显著提升推理速度
4. **保持质量**：比MQA有更多KV头，表达能力更强

## 数学公式

给定输入序列 $X \in \mathbb{R}^{n \times d}$，GQA的计算如下：

$$
\begin{aligned}
\text{GQA}(Q,K,V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, K_{g(i)}W_{g(i)}^K, V_{g(i)}W_{g(i)}^V)
\end{aligned}
$$

其中：
- $h$ 是查询头的数量
- $g(i) = \lfloor i \times G / h \rfloor$ 是头$i$所属的组
- $G$ 是键值头的数量（$G < h$）
- 每组有 $h/G$ 个查询头共享一对KV头

### 三种注意力机制对比

| 机制 | Q头数 | K头数 | V头数 | 共享方式 |
|------|-------|-------|-------|---------|
| **MHA** | h | h | h | 不共享 |
| **GQA** | h | h/G | h/G | G个Q头共享一对KV头 |
| **MQA** | h | 1 | 1 | 所有Q头共享一对KV头 |

## 详细步骤

### 1. 线性投影
对输入进行投影得到Q、K、V，其中KV的维度更小：
```python
Q = X @ W_Q  # (seq_len, embed_dim)
K = X @ W_K  # (seq_len, num_kv_heads * head_dim)
V = X @ W_V  # (seq_len, num_kv_heads * head_dim)
```

### 2. 分割成多个头
```python
# Q分割成num_heads个头
Q_heads = split_heads(Q)  # (num_heads, seq_len, head_dim)

# K,V只分割成num_kv_heads个头
K_heads = split_heads(K)  # (num_kv_heads, seq_len, head_dim)
V_heads = split_heads(V)  # (num_kv_heads, seq_len, head_dim)
```

### 3. GQA的核心：分组共享KV头
```python
for i in range(num_heads):
    # 确定当前Q头属于哪个KV组
    kv_idx = i // (num_heads // num_kv_heads)

    # 使用对应的KV头
    head_i = Attention(Q_heads[i], K_heads[kv_idx], V_heads[kv_idx])
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
 (h个头)   (G个头)  (G个头)   G < h
    |         |        |
    └────┬────┴────┬───┘
         |         |
    分组共享KV头
         |
    ┌────┼────┬────┼────┐
   组1   组2   ...  组G   (每组h/G个Q头)
    └────┼────┴────┼────┘
         |
      Concat
         |
       W^O
         |
    输出 (seq_len, embed_dim)
```

**示例：8个Q头，2个KV头**
```
Q头0, Q头1, Q头2, Q头3 → 使用 KV头0
Q头4, Q头5, Q头6, Q头7 → 使用 KV头1
```

## 代码实现

### 基础实现

```python
class GroupedQueryAttention:
    def __init__(self, embed_dim, num_heads, num_kv_heads):
        self.num_heads = num_heads        # Q头数
        self.num_kv_heads = num_kv_heads  # KV头数
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = embed_dim // num_heads

        # Q投影：完整的num_heads个头
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # K,V投影：只有num_kv_heads个头（减少参数）
        kv_dim = num_kv_heads * self.head_dim
        self.W_k = np.random.randn(embed_dim, kv_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, kv_dim) / np.sqrt(embed_dim)

        # 输出投影
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, x):
        # 1. 投影
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 2. 分割头
        Q = self.split_heads(Q, self.num_heads)
        K = self.split_heads(K, self.num_kv_heads)
        V = self.split_heads(V, self.num_kv_heads)

        # 3. 分组共享KV头
        head_outputs = []
        for i in range(self.num_heads):
            kv_idx = i // self.num_groups  # 确定使用哪个KV头
            head_out = attention(Q[i], K[kv_idx], V[kv_idx])
            head_outputs.append(head_out)

        # 4. 合并并输出投影
        concat = self.combine_heads(np.stack(head_outputs))
        output = concat @ self.W_o

        return output
```

## 参数量和KV缓存分析

假设 `embed_dim=512`, `num_heads=8`, `seq_len=1024`：

### 参数量对比

| 机制 | Q投影 | K投影 | V投影 | O投影 | 总计 |
|------|-------|-------|-------|-------|------|
| **MHA** | 512×512 | 512×512 | 512×512 | 512×512 | 1,048,576 |
| **GQA (4 KV)** | 512×512 | 512×256 | 512×256 | 512×512 | 786,432 |
| **GQA (2 KV)** | 512×512 | 512×128 | 512×128 | 512×512 | 655,360 |
| **MQA (1 KV)** | 512×512 | 512×64 | 512×64 | 512×512 | 589,824 |

### KV缓存对比

在推理时，需要缓存每个token的K和V向量：

| 机制 | KV头数 | KV缓存大小 | 相对MHA |
|------|--------|-----------|---------|
| **MHA** | 8 | 2 × 1024 × 8 × 64 = 1,048,576 | 100% |
| **GQA (4 KV)** | 4 | 2 × 1024 × 4 × 64 = 524,288 | 50% |
| **GQA (2 KV)** | 2 | 2 × 1024 × 2 × 64 = 262,144 | 25% |
| **MQA (1 KV)** | 1 | 2 × 1024 × 1 × 64 = 131,072 | 12.5% |

**关键观察**：
- GQA (2 KV头) 节省 **75%** 的KV缓存
- 参数量仅减少 **37.5%**
- 这对长序列生成至关重要！

## GQA的优势

### 1. 效率提升

**KV缓存节省**：
- 推理时的内存瓶颈主要在KV缓存
- GQA显著减少KV缓存（G倍）
- 长序列生成速度大幅提升

**计算量**：
- 参数量减少（K,V投影更小）
- 前向传播更快
- 训练和推理都受益

### 2. 质量保证

**vs MQA**：
- 更多的KV头 → 更强的表达能力
- 实验表明GQA质量接近MHA
- MQA有时会显著损失质量

**vs MHA**：
- 质量损失很小（通常<1%）
- 通过增加训练可以弥补
- 性价比极高

### 3. 实用平衡

**最佳折中**：
- 在质量和效率间找到最佳平衡点
- 不同任务可选择不同的组数
- 灵活性强，可根据需求调整

## 实际应用

### Llama 2配置

Llama 2全面采用GQA：

| 模型 | embed_dim | num_heads | num_kv_heads | 组数 |
|------|-----------|-----------|--------------|------|
| Llama 2 7B | 4096 | 32 | 8 | 4 |
| Llama 2 13B | 5120 | 40 | 8 | 5 |
| Llama 2 70B | 8192 | 64 | 8 | 8 |

**选择原因**：
- 显著减少推理时的KV缓存
- 提升长文本生成速度
- 保持接近MHA的模型质量
- 更易于部署到资源受限环境

### Mistral 7B

```python
# Mistral 7B配置
embed_dim = 4096
num_heads = 32
num_kv_heads = 8  # GQA with 4 groups
```

### 使用示例

```python
# 创建GQA层
gqa = GroupedQueryAttention(
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2  # 每4个Q头共享1对KV头
)

# 前向传播
x = np.random.randn(seq_len, 512)
output = gqa.forward(x)

print(f"输入: {x.shape}")    # (seq_len, 512)
print(f"输出: {output.shape}")  # (seq_len, 512)
```

### 带因果Mask（自回归生成）

```python
# GPT/Llama风格的自回归模型
causal_mask = np.tril(np.ones((seq_len, seq_len)))
output = gqa.forward(x, mask=causal_mask)
```

## 如何选择num_kv_heads

### 经验法则

| 场景 | 建议配置 | 原因 |
|------|---------|------|
| **质量优先** | num_kv_heads = num_heads/2 | 接近MHA质量 |
| **平衡** | num_kv_heads = num_heads/4 | 最佳性价比 |
| **效率优先** | num_kv_heads = 1 (MQA) | 最大效率 |

### 常见配置

```python
# 示例：8个查询头
num_heads = 8

# 轻度GQA（接近MHA）
num_kv_heads = 4  # 2个Q头共享1对KV

# 中度GQA（推荐）
num_kv_heads = 2  # 4个Q头共享1对KV

# 重度GQA（接近MQA）
num_kv_heads = 1  # 8个Q头共享1对KV
```

## 复杂度分析

假设序列长度为$n$，嵌入维度为$d$，Q头数为$h$，KV头数为$g$：

| 操作 | 复杂度 |
|------|--------|
| Q投影 | $O(n \cdot d^2)$ |
| K,V投影 | $O(n \cdot d \cdot g \cdot \frac{d}{h})$ |
| 注意力计算 | $O(n^2 \cdot d)$ |
| 输出投影 | $O(n \cdot d^2)$ |
| **总计** | **$O(n^2 \cdot d + n \cdot d^2)$** |

**与MHA对比**：
- 时间复杂度相同（主导项是$O(n^2 \cdot d)$）
- 参数量减少：$2d^2(1 - \frac{g}{h})$
- KV缓存减少：$(1 - \frac{g}{h}) \times 100\%$

## MHA vs GQA vs MQA对比

### 综合对比表

| 特性 | MHA | GQA | MQA |
|------|-----|-----|-----|
| **Q头数** | h | h | h |
| **KV头数** | h | g (g<h) | 1 |
| **参数量** | 最多 | 中等 | 最少 |
| **KV缓存** | 最多 | 中等 | 最少 |
| **模型质量** | 最好 | 接近MHA | 可能下降 |
| **推理速度** | 慢 | 快 | 最快 |
| **表达能力** | 最强 | 强 | 较弱 |
| **应用** | BERT, GPT-2 | Llama 2, Mistral | PaLM |

### 形象比喻

- **MHA**：每个查询头都有专属的KV秘书（8个查询头→8对KV）
- **GQA**：每2-4个查询头共享一对KV秘书（8个查询头→2对KV）
- **MQA**：所有查询头共享同一对KV秘书（8个查询头→1对KV）

## 长序列生成中的重要性

### KV缓存问题

在自回归生成中，每生成一个新token，都需要：
1. 保存该token的K和V向量
2. 使用所有历史token的KV缓存

**示例（Llama 2 7B，序列长度4096）**：

```python
# MHA配置
mha_cache = 2 × 4096 × 32 × 128 × 4bytes = 134 MB

# GQA配置 (8 KV heads)
gqa_cache = 2 × 4096 × 8 × 128 × 4bytes = 33.5 MB

# 节省：100.5 MB (75%)
```

**在批量推理时**：
- Batch size = 32
- MHA需要：4.3 GB
- GQA需要：1.1 GB
- 节省：**3.2 GB**

这使得在有限GPU内存下可以：
- 增大batch size（提高吞吐量）
- 支持更长的序列
- 部署更大的模型

## 可视化理解

### KV头共享示意图

```
num_heads = 8, num_kv_heads = 2

┌─────────────────────────────────┐
│  KV头0（被Q头0,1,2,3共享）       │
│  ┌───┬───┬───┬───┐              │
│  │Q0 │Q1 │Q2 │Q3 │ → KV0        │
│  └───┴───┴───┴───┘              │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  KV头1（被Q头4,5,6,7共享）       │
│  ┌───┬───┬───┬───┐              │
│  │Q4 │Q5 │Q6 │Q7 │ → KV1        │
│  └───┴───┴───┴───┘              │
└─────────────────────────────────┘
```

## 常见问题

### Q1: GQA比MHA慢吗？
A: 不会。虽然有分组逻辑，但KV投影更小，总体上GQA更快（尤其是推理时）。

### Q2: 如何从MHA转换到GQA？
A: 可以通过平均合并KV头来转换。Llama 2论文提供了转换方法，只需少量微调即可恢复性能。

### Q3: num_kv_heads必须整除num_heads吗？
A: 是的，必须满足 `num_heads % num_kv_heads == 0`，以确保每组Q头数量相同。

### Q4: GQA适合所有任务吗？
A: GQA特别适合长序列生成任务。对于短序列或训练阶段，MHA和GQA差异不大。

### Q5: 能否动态调整num_kv_heads？
A: 训练后不能改变。但可以训练多个配置，根据部署环境选择最合适的。

## 实现技巧

### 1. 高效的分组索引
```python
# 方法1：循环（清晰但慢）
for i in range(num_heads):
    kv_idx = i // num_groups

# 方法2：向量化（快）
kv_indices = np.arange(num_heads) // num_groups
```

### 2. 共享KV头的广播
在实际实现中，可以使用reshape和repeat来复制KV头：
```python
# K: (num_kv_heads, seq_len, head_dim)
# 复制成 (num_heads, seq_len, head_dim)
K_repeated = K.repeat(num_groups, axis=0)
```

### 3. 兼容MHA和MQA
```python
# 统一接口
if num_kv_heads is None:
    num_kv_heads = num_heads  # MHA
# num_kv_heads = 1 时自动变成MQA
```

## 参考文献

1. **GQA原论文**：Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245.

2. **Llama 2**：Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288.

3. **MQA原论文**：Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.

4. **Multi-Head Attention**：Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS.

5. **Mistral 7B**：Jiang, A. Q., et al. (2023). "Mistral 7B." arXiv:2310.06825.

## 文件说明

- `grouped_query_attention.py`: Python实现（带详细中文注释）
- `grouped_query_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档

## 总结

Grouped-Query Attention是现代大语言模型的关键优化技术，通过分组共享KV头，在保持模型质量的同时显著提升了推理效率。

**核心要点**：
- ✅ 多个Q头共享KV头，减少KV缓存
- ✅ 在质量和效率间取得最佳平衡
- ✅ 特别适合长序列生成场景
- ✅ 被Llama 2、Mistral等先进模型采用
- ✅ 是部署大模型的关键技术

**适用场景**：
- 长文本生成（小说、文章、代码）
- 对话系统（需要长上下文）
- 资源受限环境的模型部署
- 需要高吞吐量的推理服务

GQA代表了注意力机制优化的最新方向，是理解和使用现代LLM不可或缺的知识！
