# Sparse Attention (稀疏注意力)

## 概述

稀疏注意力通过只计算部分位置对的注意力来降低计算复杂度。标准注意力需要计算所有n²个位置对，而稀疏注意力通过精心设计的模式只计算有意义的连接，将复杂度从O(n²)降低到O(n×k)，其中k是稀疏度。

## 核心动机

### 标准注意力的问题

```python
# 标准注意力
Attention(Q, K, V) = softmax(QK^T) V

# 需要计算 n×n 的注意力矩阵
QK^T: (seq_len, seq_len)  # 当seq_len很大时非常昂贵
```

**问题**：
- **计算复杂度**: O(n²d)
- **内存占用**: O(n²)
- **长序列瓶颈**: 序列长度4096时，需要16M的注意力矩阵

### 核心观察

1. **局部性**: 许多任务中，远距离位置的关系较弱
2. **特殊token**: 某些位置（如[CLS]）需要全局视野
3. **长距离依赖**: 可以通过随机连接或步进模式捕获

### 稀疏注意力的解决方案

**核心思想**: 只计算有意义的位置对

```
标准注意力: 计算所有 n² 个位置对
稀疏注意力: 只计算 n×k 个位置对 (k << n)
```

**复杂度对比**:

| 方法 | 时间复杂度 | 空间复杂度 | 序列长度限制 |
|------|-----------|-----------|------------|
| 标准注意力 | O(n²d) | O(n²) | ~512 |
| 稀疏注意力 | O(n×k×d) | O(n×k) | ~4096+ |

## 稀疏模式类型

### 1. Local (局部窗口)

每个位置只关注前后窗口范围内的邻居。

```
模式: 带状对角矩阵
复杂度: O(n×w×d)，其中w是窗口大小
```

**可视化**:
```
    0 1 2 3 4 5 6 7
0 [ 1 1 1 0 0 0 0 0 ]
1 [ 1 1 1 1 0 0 0 0 ]
2 [ 1 1 1 1 1 0 0 0 ]
3 [ 0 1 1 1 1 1 0 0 ]
4 [ 0 0 1 1 1 1 1 0 ]
...
```

**适用场景**:
- 语言建模（相邻词语关系强）
- 局部结构明显的任务

**优点**:
- ✅ 实现简单
- ✅ 稀疏度高
- ✅ 适合局部依赖

**缺点**:
- ❌ 难以捕获长距离依赖

### 2. Global (全局token)

特定位置可以关注和被关注所有位置。

```
模式: 特定行/列全为1
复杂度: O(n×(w+g)×d)，g是全局token数
```

**可视化**:
```
    0 1 2 3 4 5 6 7
0 [ 1 1 1 1 1 1 1 1 ]  ← 全局行
1 [ 1 1 1 0 0 0 0 0 ]
2 [ 1 1 1 1 0 0 0 0 ]
3 [ 1 0 1 1 1 0 0 0 ]
    ↑ 全局列
```

**适用场景**:
- 分类任务（[CLS] token）
- 需要汇总全局信息

**优点**:
- ✅ 保留全局视野
- ✅ 适合特殊token

**缺点**:
- ❌ 全局token成为计算瓶颈

### 3. Random (随机连接)

每个位置随机关注k个其他位置。

```
模式: 随机稀疏矩阵
复杂度: O(n×r×d)，r是随机连接数
```

**适用场景**:
- 捕获长距离依赖
- 作为其他模式的补充

**优点**:
- ✅ 覆盖全序列
- ✅ 打破局部限制

**缺点**:
- ❌ 不确定性
- ❌ 难以复现

### 4. Strided (步进)

按固定步长采样位置。

```
模式: 等间距连接
复杂度: O(n×(n/s)×d)，s是步长
```

**可视化** (步长=2):
```
    0 1 2 3 4 5 6 7
0 [ 1 0 1 0 1 0 1 0 ]
1 [ 0 1 0 1 0 1 0 1 ]
2 [ 1 0 1 0 1 0 1 0 ]
...
```

**适用场景**:
- 层次化结构
- 多尺度特征

### 5. Longformer (局部+全局)

组合局部窗口和全局token。

**公式**:
```
Mask_Longformer = Mask_Local ∪ Mask_Global
```

**特点**:
- 大部分位置使用局部窗口
- 特殊位置（[CLS]、[SEP]）使用全局注意力

**稀疏度**: 约 `(2w+1)/n + 2g/n`

**应用**: Longformer论文，适合文档理解

### 6. BigBird (局部+全局+随机)

组合三种模式：局部窗口、全局token、随机连接。

**公式**:
```
Mask_BigBird = Mask_Local ∪ Mask_Global ∪ Mask_Random
```

**特点**:
- 局部窗口：捕获局部依赖
- 全局token：汇总全局信息
- 随机连接：捕获长距离依赖

**理论保证**: BigBird论文证明这种组合可以近似全注意力。

**稀疏度**: 约 `(2w+1+r)/n + 2g/n`

**应用**: BigBird论文，适合长文档、基因组序列

## 实现细节

### 基本实现

```python
class SparseAttention:
    def __init__(self, embed_dim, pattern='local', **kwargs):
        self.embed_dim = embed_dim
        self.pattern = pattern
        self.kwargs = kwargs

        # 初始化Q、K、V投影
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def create_mask(self, seq_len):
        """根据pattern创建稀疏mask"""
        if self.pattern == 'local':
            return create_local_mask(seq_len, self.kwargs['window_size'])
        # ... 其他模式

    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 创建稀疏mask
        mask = self.create_mask(len(x))

        # 计算注意力（mask掉不需要的位置）
        scores = (Q @ K.T) / sqrt(embed_dim)
        scores = np.where(mask == 1, scores, -1e9)

        attention = softmax(scores)
        output = attention @ V

        return output
```

### 局部窗口实现

```python
def create_local_mask(seq_len, window_size):
    """
    创建局部窗口mask

    每个位置i关注 [i-window_size, i+window_size] 范围
    """
    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1

    return mask
```

### Longformer实现

```python
def create_longformer_mask(seq_len, window_size, global_indices):
    """
    Longformer: 局部窗口 + 全局token
    """
    # 局部窗口
    local_mask = create_local_mask(seq_len, window_size)

    # 全局token
    global_mask = np.zeros((seq_len, seq_len))
    for idx in global_indices:
        global_mask[idx, :] = 1  # 全局位置关注所有
        global_mask[:, idx] = 1  # 所有位置关注全局

    # 组合
    mask = np.maximum(local_mask, global_mask)

    return mask
```

### BigBird实现

```python
def create_bigbird_mask(seq_len, window_size, num_random, global_indices):
    """
    BigBird: 局部 + 全局 + 随机
    """
    # 局部
    local_mask = create_local_mask(seq_len, window_size)

    # 全局
    global_mask = create_global_mask(seq_len, global_indices)

    # 随机
    random_mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        random_indices = np.random.choice(seq_len, size=num_random, replace=False)
        random_mask[i, random_indices] = 1

    # 组合所有模式
    mask = np.maximum(np.maximum(local_mask, global_mask), random_mask)

    return mask
```

## 性能分析

### 稀疏度对比

以序列长度512、窗口大小5为例：

| 模式 | 稀疏度 | 活跃连接数 | 加速比 |
|------|-------|----------|--------|
| Full (标准) | 100% | 262,144 | 1.0x |
| Local (w=5) | 2.1% | 5,632 | ~47x |
| Longformer | 2.5% | 6,656 | ~39x |
| BigBird | 3.4% | 8,960 | ~29x |

### 不同序列长度的稀疏度

| 序列长度 | Local | Longformer | BigBird |
|---------|-------|-----------|---------|
| 64      | 17.2% | 20.3% | 25.0% |
| 128     | 8.6%  | 10.2% | 12.9% |
| 256     | 4.3%  | 5.1%  | 6.6%  |
| 512     | 2.1%  | 2.5%  | 3.4%  |
| 1024    | 1.1%  | 1.3%  | 1.8%  |
| 2048    | 0.5%  | 0.6%  | 0.9%  |

**观察**: 序列越长，稀疏度越低，加速效果越明显。

### 计算复杂度对比

**理论FLOPs** (序列长度n, 窗口大小w, 嵌入维度d):

| 模式 | 复杂度 | n=512, d=64 | n=2048, d=64 |
|------|-------|------------|-------------|
| Full | O(n²d) | 1.07 GFLOPs | 17.2 GFLOPs |
| Local | O(nwd) | 0.023 GFLOPs | 0.09 GFLOPs |
| Longformer | O(n(w+g)d) | 0.026 GFLOPs | 0.10 GFLOPs |
| BigBird | O(n(w+g+r)d) | 0.035 GFLOPs | 0.14 GFLOPs |

### 窗口大小的影响

固定序列长度512：

| 窗口大小 | 稀疏度 | 活跃连接数 | 相对FLOPs |
|---------|-------|----------|----------|
| 1       | 0.6%  | 1,536    | 0.6%     |
| 3       | 1.4%  | 3,584    | 1.4%     |
| 5       | 2.1%  | 5,632    | 2.1%     |
| 10      | 4.1%  | 10,752   | 4.1%     |
| 20      | 8.0%  | 20,992   | 8.0%     |

**权衡**: 窗口越大，性能越好但计算量越大。

## 使用示例

### 基本用法

```python
from sparse_attention import SparseAttention
import numpy as np

# 创建稀疏注意力层
embed_dim = 64
sparse_attn = SparseAttention(
    embed_dim,
    pattern='local',
    window_size=5
)

# 前向传播
x = np.random.randn(512, embed_dim)
output, mask, sparsity = sparse_attn.forward(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"稀疏度: {sparsity:.2%}")
```

### Longformer风格

```python
# 文档分类任务
longformer_attn = SparseAttention(
    embed_dim,
    pattern='longformer',
    window_size=5,
    global_indices=[0]  # [CLS] token
)

output, mask, sparsity = longformer_attn.forward(x)
print(f"Longformer稀疏度: {sparsity:.2%}")
```

### BigBird风格

```python
# 长文档理解
bigbird_attn = SparseAttention(
    embed_dim,
    pattern='bigbird',
    window_size=5,
    num_random=3,
    global_indices=[0, -1]  # [CLS] 和 [SEP]
)

output, mask, sparsity = bigbird_attn.forward(x)
print(f"BigBird稀疏度: {sparsity:.2%}")
```

### 可视化稀疏模式

```python
from sparse_attention import visualize_sparse_patterns

# 生成并可视化不同的稀疏模式
visualize_sparse_patterns(seq_len=64)
```

## 应用场景

### 适合使用稀疏注意力的场景

1. **长文档处理**
   - 法律文书分析（数千词）
   - 学术论文理解
   - 书籍摘要

2. **长序列任务**
   - 基因组序列分析（数万碱基对）
   - 长视频理解（数千帧）
   - 长时序预测

3. **资源受限场景**
   - 移动设备部署
   - 边缘计算
   - 大批量处理

### 不同模式的选择

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 语言建模 | Local | 相邻词语关系最重要 |
| 文档分类 | Longformer | 需要[CLS]汇总全局信息 |
| 长文档QA | BigBird | 平衡局部、全局和长距离 |
| 图像生成 | Strided | 多尺度特征 |

## 相关工作

### 1. Sparse Transformer (OpenAI, 2019)

**论文**: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)

**贡献**:
- 首次系统研究稀疏注意力模式
- 提出Factorized attention
- 应用于图像和音乐生成

**模式**:
- Strided attention
- Fixed attention

### 2. Longformer (AllenAI, 2020)

**论文**: [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

**贡献**:
- 局部窗口 + 全局token
- 高效的CUDA实现
- 处理最长16,384 tokens

**应用**:
- 长文档分类
- 文档问答
- 摘要生成

### 3. BigBird (Google Research, 2020)

**论文**: [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)

**贡献**:
- 局部 + 全局 + 随机三种模式
- 理论证明可以近似全注意力
- 处理最长8,192 tokens

**理论**:
- 证明稀疏注意力是图灵完备的
- 提供近似保证

**应用**:
- 基因组序列
- 长文档摘要
- 问答系统

### 4. ETC (Google, 2020)

**Extended Transformer Construction**

**特点**:
- Structured attention
- Global-local attention
- 相对位置编码

## 实现优化

### 1. 高效稀疏矩阵操作

```python
# 使用稀疏矩阵库
from scipy.sparse import csr_matrix

# 只存储非零元素
sparse_scores = csr_matrix(scores * mask)
```

### 2. 分块计算

```python
# 对于超长序列，分块处理
def chunked_sparse_attention(x, chunk_size=512):
    outputs = []
    for i in range(0, len(x), chunk_size):
        chunk = x[i:i+chunk_size]
        output = sparse_attention(chunk)
        outputs.append(output)
    return np.concatenate(outputs)
```

### 3. CUDA优化

实际应用中，需要CUDA kernel优化：
- 只计算mask为1的位置
- 使用共享内存
- 并行化稀疏矩阵乘法

Longformer和BigBird都提供了高效的CUDA实现。

## 优势与限制

### 优势

1. **计算效率**: 显著降低复杂度
2. **内存节省**: O(n²) → O(n×k)
3. **可扩展性**: 支持更长序列
4. **灵活性**: 可以设计任务相关的模式
5. **理论保证**: BigBird证明了近似能力

### 限制

1. **设计复杂**: 需要精心选择稀疏模式
2. **任务相关**: 不是所有任务都适合稀疏
3. **实现难度**: 高效实现需要定制CUDA kernel
4. **精度损失**: 某些任务可能需要密集注意力

## 文件说明

- `sparse_attention.py`: 完整实现，包含多种稀疏模式
- `sparse_attention.ipynb`: 交互式教程，带可视化
- `README.md`: 本文档

## 运行示例

```bash
# 运行Python脚本
python sparse_attention.py

# 或使用Jupyter Notebook
jupyter notebook sparse_attention.ipynb
```

## 参考资料

1. [Sparse Transformer](https://arxiv.org/abs/1904.10509) - OpenAI, 2019
2. [Longformer](https://arxiv.org/abs/2004.05150) - AllenAI, 2020
3. [Big Bird](https://arxiv.org/abs/2007.14062) - Google Research, 2020
4. [ETC](https://arxiv.org/abs/2004.08483) - Google, 2020

## 总结

稀疏注意力通过精心设计的模式，在保持性能的同时显著降低了计算复杂度，使得处理超长序列成为可能。

**核心优势**:
- ✅ 计算复杂度 O(n²) → O(n×k)
- ✅ 内存占用大幅降低
- ✅ 支持超长序列（4k-16k tokens）
- ✅ 多种模式可选

**适用场景**:
- 长文档处理
- 基因组序列
- 长视频理解
- 资源受限场景

**模式选择**:
- **Local**: 局部依赖强
- **Longformer**: 需要全局信息
- **BigBird**: 平衡各种依赖
