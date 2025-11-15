# Sliding Window Attention (滑动窗口注意力)

## 概述

Sliding Window Attention (滑动窗口注意力) 是一种高效的注意力机制，每个token只关注固定窗口内的邻近token，而不是全序列的所有token。这种方法由Beltagy等人在Longformer论文中系统化推广，并在Mistral、BigBird等现代长序列模型中被广泛采用。

滑动窗口注意力通过限制注意力范围，将计算复杂度从O(n²)降低到O(n×w)，使得处理超长序列成为可能。

## 核心思想

滑动窗口注意力的关键特点：

1. **局部注意力**：每个token只关注固定窗口内的token
2. **线性复杂度**：计算量与序列长度线性相关O(n×w)
3. **稀疏模式**：注意力矩阵呈现带状稀疏结构
4. **可扩展性**：可以处理数万甚至更长的序列

## 数学公式

### 标准注意力（全局）

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

计算复杂度：$O(n^2 \cdot d)$

### 滑动窗口注意力

对于位置$i$，只计算窗口内位置的注意力：

$$
\text{Attention}_i = \text{softmax}\left(\frac{Q_i \cdot K_{[i-w:i+w]}}{\sqrt{d_k}}\right) \cdot V_{[i-w:i+w]}
$$

其中：
- $w$ 是窗口大小（每侧的范围）
- $[i-w:i+w]$ 表示位置$i$的窗口范围
- 窗口总大小为$2w+1$（包括位置$i$自己）

计算复杂度：$O(n \cdot w \cdot d)$

### 通过Mask实现

实际上，滑动窗口通过mask矩阵实现：

$$
\begin{aligned}
\text{Scores} &= \frac{QK^T}{\sqrt{d_k}} \\
\text{Masked Scores} &= \text{Scores} \odot \text{WindowMask} \\
\text{Attention} &= \text{softmax}(\text{Masked Scores}) \cdot V
\end{aligned}
$$

其中WindowMask是带状矩阵：

$$
\text{WindowMask}[i,j] = \begin{cases}
1 & \text{if } |i-j| \leq w \\
0 & \text{otherwise}
\end{cases}
$$

## 详细步骤

### 1. 创建滑动窗口Mask

```python
def create_sliding_window_mask(seq_len, window_size):
    """创建滑动窗口mask"""
    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        # 计算窗口范围
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1

    return mask
```

**示例（seq_len=8, window_size=2）**：

```
[[1 1 1 0 0 0 0 0]    位置0: 可见[0,1,2]
 [1 1 1 1 0 0 0 0]    位置1: 可见[0,1,2,3]
 [1 1 1 1 1 0 0 0]    位置2: 可见[0,1,2,3,4]
 [0 1 1 1 1 1 0 0]    位置3: 可见[1,2,3,4,5]
 [0 0 1 1 1 1 1 0]    位置4: 可见[2,3,4,5,6]
 [0 0 0 1 1 1 1 1]    位置5: 可见[3,4,5,6,7]
 [0 0 0 0 1 1 1 1]    位置6: 可见[4,5,6,7]
 [0 0 0 0 0 1 1 1]]   位置7: 可见[5,6,7]
```

### 2. 应用窗口Mask到注意力计算

```python
# 步骤1：计算注意力得分
scores = Q @ K.T / sqrt(d_k)  # (seq_len, seq_len)

# 步骤2：创建窗口mask
window_mask = create_sliding_window_mask(seq_len, window_size)

# 步骤3：将窗口外的位置设为-inf
scores = np.where(window_mask == 0, -1e9, scores)

# 步骤4：Softmax（窗口外的位置权重为0）
attention_weights = softmax(scores)

# 步骤5：加权求和
output = attention_weights @ V
```

## 架构图示

### 滑动窗口模式

```
                      Key序列
         0   1   2   3   4   5   6   7
      ┌───┬───┬───┬───┬───┬───┬───┬───┐
   0  │ ■ │ ■ │ ■ │   │   │   │   │   │
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   1  │ ■ │ ■ │ ■ │ ■ │   │   │   │   │
      ├───┼───┼───┼───┼───┼───┼───┼───┤
Q  2  │ ■ │ ■ │ ■ │ ■ │ ■ │   │   │   │  窗口大小=2
u     ├───┼───┼───┼───┼───┼───┼───┼───┤
e  3  │   │ ■ │ ■ │ ■ │ ■ │ ■ │   │   │  ■ = 可见
r     ├───┼───┼───┼───┼───┼───┼───┼───┤  空 = 被mask
y  4  │   │   │ ■ │ ■ │ ■ │ ■ │ ■ │   │
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   5  │   │   │   │ ■ │ ■ │ ■ │ ■ │ ■ │
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   6  │   │   │   │   │ ■ │ ■ │ ■ │ ■ │
      ├───┼───┼───┼───┼───┼───┼───┼───┤
   7  │   │   │   │   │   │ ■ │ ■ │ ■ │
      └───┴───┴───┴───┴───┴───┴───┴───┘

特点：带状稀疏模式，沿对角线分布
```

### 对比标准注意力

```
标准注意力（密集）         滑动窗口注意力（稀疏）
┌─────────────┐           ┌─────────────┐
│■ ■ ■ ■ ■ ■ ■│           │■ ■ ■       │
│■ ■ ■ ■ ■ ■ ■│           │■ ■ ■ ■     │
│■ ■ ■ ■ ■ ■ ■│           │■ ■ ■ ■ ■   │
│■ ■ ■ ■ ■ ■ ■│  vs       │  ■ ■ ■ ■ ■ │
│■ ■ ■ ■ ■ ■ ■│           │    ■ ■ ■ ■ ■│
│■ ■ ■ ■ ■ ■ ■│           │      ■ ■ ■ ■│
│■ ■ ■ ■ ■ ■ ■│           │        ■ ■ ■│
└─────────────┘           └─────────────┘
  O(n²) 复杂度               O(n×w) 复杂度
```

## 代码实现

### 基础实现

```python
class SlidingWindowAttention:
    def __init__(self, embed_dim, num_heads, window_size):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        # Q, K, V投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, x):
        seq_len = x.shape[0]

        # 1. 线性投影
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 2. 分割头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算注意力得分
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim)

        # 4. 应用滑动窗口mask
        window_mask = create_sliding_window_mask(seq_len, self.window_size)
        scores = np.where(window_mask[None, :, :] == 0, -1e9, scores)

        # 5. Softmax
        attention_weights = softmax(scores)

        # 6. 加权求和
        output = attention_weights @ V

        # 7. 合并头并输出投影
        output = self.combine_heads(output)
        output = output @ self.W_o

        return output
```

## 复杂度分析

### 时间复杂度

| 操作 | 标准注意力 | 滑动窗口注意力 |
|------|-----------|---------------|
| **Q,K,V投影** | $O(n \cdot d^2)$ | $O(n \cdot d^2)$ |
| **注意力得分** | $O(n^2 \cdot d)$ | $O(n \cdot w \cdot d)$ |
| **Softmax** | $O(n^2)$ | $O(n \cdot w)$ |
| **加权求和** | $O(n^2 \cdot d)$ | $O(n \cdot w \cdot d)$ |
| **输出投影** | $O(n \cdot d^2)$ | $O(n \cdot d^2)$ |
| **总计** | **$O(n^2 \cdot d + n \cdot d^2)$** | **$O(n \cdot w \cdot d + n \cdot d^2)$** |

**关键观察**：
- 当序列长度$n \gg d$时，注意力计算主导复杂度
- 标准注意力：$O(n^2 \cdot d)$（二次复杂度）
- 滑动窗口：$O(n \cdot w \cdot d)$（线性复杂度）
- 加速比：$\frac{n^2}{n \cdot w} = \frac{n}{w}$

### 空间复杂度

| 组件 | 标准注意力 | 滑动窗口注意力 |
|------|-----------|---------------|
| **注意力矩阵** | $O(n^2)$ | $O(n \cdot w)$ |
| **中间结果** | $O(n \cdot d)$ | $O(n \cdot d)$ |
| **总计** | **$O(n^2 + n \cdot d)$** | **$O(n \cdot w + n \cdot d)$** |

### 实际性能对比

假设$d=512$, $w=128$：

| 序列长度 | 标准注意力操作数 | 滑动窗口操作数 | 加速比 | 内存节省 |
|---------|-----------------|---------------|--------|---------|
| 128 | 16,384 | 16,384 | 1.0x | 0% |
| 512 | 262,144 | 65,536 | 4.0x | 75% |
| 1,024 | 1,048,576 | 131,072 | 8.0x | 87.5% |
| 4,096 | 16,777,216 | 524,288 | 32.0x | 96.9% |
| 16,384 | 268,435,456 | 2,097,152 | 128.0x | 99.2% |

**关键结论**：序列越长，滑动窗口的优势越明显！

## 优势与局限

### 优势

1. **计算效率极高**
   - 线性复杂度O(n×w)
   - 序列长度翻倍，计算量仅翻倍
   - 标准注意力：序列长度翻倍，计算量翻4倍

2. **内存占用低**
   - 只需存储窗口内的注意力权重
   - 稀疏矩阵可以高效存储
   - 适合GPU内存受限的场景

3. **可扩展到超长序列**
   - Mistral处理32K长度序列
   - Longformer处理16K长度文档
   - 理论上可以处理更长序列

4. **局部建模能力强**
   - 捕获邻近依赖关系
   - 适合语言、视频、音频等序列
   - 局部模式建模效果好

### 局限

1. **全局信息传递受限**
   - 窗口外的token无法直接交互
   - 需要多层堆叠扩大感受野
   - 远距离依赖建模较弱

2. **可能需要混合策略**
   - Longformer：全局token + 滑动窗口
   - BigBird：随机注意力 + 滑动窗口 + 全局token
   - 增加实现复杂度

3. **窗口大小需要调优**
   - 窗口太小：上下文不足
   - 窗口太大：效率降低
   - 需要根据任务选择

## 实际应用

### Mistral 7B

Mistral 7B全面使用滑动窗口注意力：

```python
# Mistral 7B配置
embed_dim = 4096
num_heads = 32
window_size = 4096      # 滑动窗口大小
max_seq_len = 32768     # 支持的序列长度
```

**关键特点**：
- 窗口大小4096提供充足的局部上下文
- 可以处理32K长度的序列
- 比标准注意力快8倍（32K/4K）
- 适合长对话和文档处理

**性能数据**：
- 序列长度32K时，标准注意力：1B操作
- 滑动窗口（w=4K）：268M操作
- 加速比：3.7x
- 内存节省：73%

### Longformer

Longformer使用混合注意力策略：

```python
# Longformer配置
embed_dim = 768
num_heads = 12
window_size = 512       # 局部窗口
global_attention = True # 部分token使用全局注意力
max_seq_len = 4096
```

**混合策略**：

1. **滑动窗口注意力**（大部分token）
   - 局部建模，高效计算
   - 每个token看到前后512个位置

2. **全局注意力**（少数特殊token）
   - [CLS], [SEP]等特殊token
   - 任务相关的关键token（如问题中的token）
   - 可以看到所有位置并被所有位置看到

3. **扩张窗口**（可选）
   - 在某些层使用更大的窗口
   - 逐层扩大感受野

**应用场景**：
- 长文档分类
- 文档问答
- 摘要生成

### BigBird

BigBird组合多种稀疏注意力模式：

```python
# BigBird配置
embed_dim = 768
num_heads = 12
window_size = 3         # 滑动窗口（每侧3个）
num_random_blocks = 3   # 随机注意力块数
num_global_tokens = 2   # 全局token数（如[CLS]）
```

**三种注意力组合**：

1. **滑动窗口**：局部依赖
2. **随机注意力**：远距离连接
3. **全局token**：全局信息聚合

**优势**：
- 平衡局部和全局建模
- 理论上近似全注意力
- 适合多种长序列任务

### 使用示例

```python
# 创建滑动窗口注意力层
swa = SlidingWindowAttention(
    embed_dim=512,
    num_heads=8,
    window_size=128
)

# 前向传播
x = np.random.randn(seq_len, 512)
output = swa.forward(x)

print(f"输入: {x.shape}")    # (seq_len, 512)
print(f"输出: {output.shape}")  # (seq_len, 512)
```

### 自回归生成（单侧窗口）

```python
# 创建单侧窗口mask（只看左侧）
def create_sliding_window_mask_one_sided(seq_len, window_size):
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = i + 1  # 包括自己
        mask[i, start:end] = 1
    return mask

# 使用单侧窗口
one_sided_mask = create_sliding_window_mask_one_sided(seq_len, window_size)
output = swa.forward(x, window_mask=one_sided_mask)
```

**用途**：
- GPT风格的自回归语言模型
- 只能看到历史信息
- 结合滑动窗口和因果约束

## 如何选择窗口大小

### 经验法则

| 场景 | 建议窗口大小 | 原因 |
|------|-------------|------|
| **短文本** | 32-128 | 覆盖大部分序列 |
| **中等文本** | 256-512 | 平衡效率和上下文 |
| **长文档** | 512-1024 | 充足的局部上下文 |
| **超长序列** | 2048-4096 | Mistral等模型的选择 |

### 权衡考虑

1. **任务特性**
   - 局部任务（如NER）：较小窗口即可
   - 需要长距离依赖：较大窗口或混合策略

2. **序列长度**
   - 窗口应小于序列长度
   - 通常为序列长度的10-25%

3. **计算资源**
   - 窗口越大，计算越慢
   - 需要在效率和效果间平衡

4. **层数**
   - 更多层可以用较小窗口
   - 通过堆叠扩大感受野

### 代码示例

```python
# 根据序列长度动态选择窗口
def adaptive_window_size(seq_len):
    if seq_len <= 512:
        return min(seq_len // 4, 128)
    elif seq_len <= 4096:
        return 512
    else:
        return 1024

# 使用
window_size = adaptive_window_size(seq_len)
```

## 扩大感受野的方法

### 1. 多层堆叠

```
层1: 窗口大小w → 感受野 = 2w+1
层2: 窗口大小w → 感受野 = 2(2w+1) + 2w+1 = 6w+3
层3: 窗口大小w → 感受野 = 14w+7
...
层n: 感受野 ≈ O(2^n × w)
```

**示例**（w=128，12层）：
- 1层：257个token
- 6层：8,065个token
- 12层：524,161个token

**结论**：通过堆叠，滑动窗口可以建模长距离依赖！

### 2. 扩张窗口

```python
# 不同层使用不同窗口大小
layer_configs = [
    {'window_size': 64},   # 层1
    {'window_size': 128},  # 层2
    {'window_size': 256},  # 层3
    {'window_size': 512},  # 层4
]
```

### 3. 混合策略

```python
# Longformer风格
mixed_mask = sliding_window_mask | global_token_mask | random_mask
```

## 可视化理解

### 窗口覆盖范围

```
序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
窗口大小 = 2

位置0的窗口: [0, 1, 2]          ←→
位置1的窗口:    [0, 1, 2, 3]       ←→
位置5的窗口:             [3, 4, 5, 6, 7]    ←→
位置9的窗口:                      [7, 8, 9] ←→
```

### 多层感受野扩展

```
层1 (窗口=1):
位置4 → [3, 4, 5]

层2 (窗口=1):
位置4 → [2, 3, 4, 5, 6]  (通过层1的传播)

层3 (窗口=1):
位置4 → [1, 2, 3, 4, 5, 6, 7]
```

## 常见问题

### Q1: 滑动窗口会损失性能吗？
A: 在长序列任务上通常不会。Mistral、Longformer等模型证明，合理的窗口大小可以保持甚至提升性能，同时大幅提升效率。

### Q2: 如何处理全局信息？
A: 三种方法：
1. 堆叠多层（推荐）
2. 混合全局token（Longformer）
3. 添加随机或其他注意力模式（BigBird）

### Q3: 窗口大小如何选择？
A: 经验值：
- 短序列(<1K): 128-256
- 中等序列(1-4K): 512
- 长序列(>4K): 1024-4096

### Q4: 可以用于编码器和解码器吗？
A: 可以！
- 编码器：双向窗口
- 解码器：单侧窗口（因果）

### Q5: 如何实现高效的滑动窗口？
A: 使用稀疏矩阵库（如PyTorch的torch.sparse）或专门的CUDA kernel优化。

## 实现技巧

### 1. 稀疏矩阵存储

```python
# 只存储非零元素
from scipy.sparse import dia_matrix

# 创建带状矩阵
diagonals = [np.ones(seq_len) for _ in range(-window_size, window_size+1)]
offsets = list(range(-window_size, window_size+1))
sparse_mask = dia_matrix((diagonals, offsets), shape=(seq_len, seq_len))
```

### 2. 向量化窗口操作

```python
# 避免循环，使用向量化操作
indices = np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]
window_mask = (np.abs(indices) <= window_size).astype(float)
```

### 3. 缓存窗口Mask

```python
# 预计算常用长度的mask
class WindowMaskCache:
    def __init__(self, window_size):
        self.window_size = window_size
        self.cache = {}

    def get_mask(self, seq_len):
        if seq_len not in self.cache:
            self.cache[seq_len] = create_sliding_window_mask(seq_len, self.window_size)
        return self.cache[seq_len]
```

## 参考文献

1. **Longformer**：Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The Long-Document Transformer." arXiv:2004.05150.

2. **BigBird**：Zaheer, M., et al. (2020). "Big Bird: Transformers for Longer Sequences." NeurIPS.

3. **Mistral 7B**：Jiang, A. Q., et al. (2023). "Mistral 7B." arXiv:2310.06825.

4. **Sparse Transformers**：Child, R., et al. (2019). "Generating Long Sequences with Sparse Transformers." arXiv:1904.10509.

5. **ETC**：Ainslie, J., et al. (2020). "ETC: Encoding Long and Structured Inputs in Transformers." EMNLP.

## 文件说明

- `sliding_window_attention.py`: Python实现（带详细中文注释）
- `sliding_window_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档

## 总结

Sliding Window Attention通过限制注意力范围到局部窗口，在保持局部建模能力的同时大幅降低计算复杂度，是处理长序列的关键技术。

**核心要点**：
- ✅ 线性复杂度O(n×w)，可扩展到超长序列
- ✅ 内存占用低，稀疏注意力模式
- ✅ 局部建模能力强，捕获邻近依赖
- ✅ 被Mistral、Longformer等先进模型采用
- ✅ 可以通过堆叠扩大感受野

**适用场景**：
- 长文档处理（数千到数万token）
- 长对话系统
- 视频/音频序列建模
- 任何需要处理长序列的场景

**设计建议**：
- 窗口大小通常为序列长度的10-25%
- Mistral的4096窗口是良好的起点
- 考虑混合策略以增强全局建模
- 堆叠多层以扩大感受野

滑动窗口注意力是突破长序列瓶颈的核心技术，为大规模语言模型处理长文本提供了高效的解决方案！
