# Relative Position Attention (相对位置注意力)

## 概述

Relative Position Attention (相对位置注意力) 是一种改进的注意力机制，它编码token之间的相对位置关系，而非绝对位置。由Shaw等人在2018年的论文"Self-Attention with Relative Position Representations"中提出，并在T5、Transformer-XL、DeBERTa等模型中被广泛采用。

相对位置注意力通过关注token之间的相对距离，提供了更好的长度泛化能力和更自然的语言建模方式。

## 核心思想

相对位置注意力的关键特点：

1. **相对位置编码**：使用相对位置而非绝对位置
2. **位置不变性**：注意力模式只依赖于相对距离
3. **长度泛化**：可以处理比训练时更长的序列
4. **参数效率**：使用max_relative_position限制参数量

## 数学公式

### 标准注意力（绝对位置）

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 相对位置注意力

对于位置$i$和$j$，注意力得分计算为：

$$
\begin{aligned}
\text{score}(i,j) &= \frac{Q_i \cdot K_j + Q_i \cdot R_{i-j}}{\sqrt{d_k}} \\
&= \frac{\text{内容得分} + \text{相对位置得分}}{\sqrt{d_k}}
\end{aligned}
$$

其中：
- $Q_i$ 是第$i$个位置的查询向量
- $K_j$ 是第$j$个位置的键向量
- $R_{i-j}$ 是相对位置嵌入，**只依赖于相对距离$i-j$**
- $d_k$ 是键向量的维度

完整的注意力计算：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T + \text{PositionScores}}{\sqrt{d_k}}\right)V
$$

其中PositionScores矩阵的第$(i,j)$个元素为$Q_i \cdot R_{i-j}$。

## 详细步骤

### 1. 生成相对位置矩阵

首先计算所有位置对之间的相对距离：

```python
# 对于序列长度seq_len，生成相对位置矩阵
range_vec = np.arange(seq_len)
relative_positions = range_vec[None, :] - range_vec[:, None]

# 裁剪到[-max_relative_position, max_relative_position]
relative_positions = np.clip(
    relative_positions,
    -max_relative_position,
    max_relative_position
)
```

**示例（seq_len=4）**：
```
[[ 0, -1, -2, -3],
 [ 1,  0, -1, -2],
 [ 2,  1,  0, -1],
 [ 3,  2,  1,  0]]
```

### 2. 相对位置到索引的映射

将相对位置映射到嵌入索引：

```python
# 相对位置[-max, +max]映射到索引[0, 2*max]
indices = relative_positions + max_relative_position
```

### 3. 计算注意力得分

```python
# 步骤1：内容得分
content_scores = Q @ K.T  # (seq_len, seq_len)

# 步骤2：相对位置得分
position_scores = np.zeros((seq_len, seq_len))
for i in range(seq_len):
    for j in range(seq_len):
        rel_idx = relative_indices[i, j]
        position_scores[i, j] = Q[i] @ relative_embeddings[rel_idx]

# 步骤3：合并得分
scores = (content_scores + position_scores) / sqrt(d_k)

# 步骤4：Softmax
attention_weights = softmax(scores)

# 步骤5：加权求和
output = attention_weights @ V
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
    计算注意力得分
         |
    ┌────┴────┐
    |         |
 内容得分  位置得分
(Q·K^T)  (Q·R_{i-j})
    |         |
    └────┬────┘
         |
      合并 & Softmax
         |
    加权求和 V
         |
    输出 (seq_len, embed_dim)
```

### 相对位置矩阵可视化

对于seq_len=8, max_relative_position=4：

```
Query位置 →
  0   1   2   3   4   5   6   7
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │-1 │-2 │-3 │-4 │-4 │-4 │-4 │ 0  ↓
├───┼───┼───┼───┼───┼───┼───┼───┤    K
│ 1 │ 0 │-1 │-2 │-3 │-4 │-4 │-4 │ 1  e
├───┼───┼───┼───┼───┼───┼───┼───┤    y
│ 2 │ 1 │ 0 │-1 │-2 │-3 │-4 │-4 │ 2  位
├───┼───┼───┼───┼───┼───┼───┼───┤    置
│ 3 │ 2 │ 1 │ 0 │-1 │-2 │-3 │-4 │ 3
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 4 │ 3 │ 2 │ 1 │ 0 │-1 │-2 │-3 │ 4
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 4 │ 4 │ 3 │ 2 │ 1 │ 0 │-1 │-2 │ 5
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 4 │ 4 │ 4 │ 3 │ 2 │ 1 │ 0 │-1 │ 6
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 4 │ 4 │ 4 │ 4 │ 3 │ 2 │ 1 │ 0 │ 7
└───┴───┴───┴───┴───┴───┴───┴───┘

说明：超过±4的距离被裁剪为±4
```

## 代码实现

### 基础实现

```python
class RelativePositionAttention:
    def __init__(self, embed_dim, num_heads, max_relative_position=None):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_position = max_relative_position

        # Q, K, V投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # 相对位置嵌入
        if max_relative_position is not None:
            num_relative_positions = 2 * max_relative_position + 1
            self.relative_position_embeddings = np.random.randn(
                num_heads, num_relative_positions, self.head_dim
            ) / np.sqrt(self.head_dim)

    def forward(self, x):
        # 1. 线性投影
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 2. 分割头
        Q = self.split_heads(Q)  # (num_heads, seq_len, head_dim)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算内容得分
        content_scores = Q @ K.transpose(0, 2, 1)

        # 4. 计算相对位置得分
        position_scores = self.compute_position_scores(Q)

        # 5. 合并得分并计算注意力
        scores = (content_scores + position_scores) / np.sqrt(self.head_dim)
        attention_weights = softmax(scores)

        # 6. 加权求和
        output = attention_weights @ V

        # 7. 合并头并输出投影
        output = self.combine_heads(output)
        output = output @ self.W_o

        return output
```

## 参数量分析

假设 `embed_dim=512`, `num_heads=8`, `max_relative_position=32`：

### 参数量构成

| 组件 | 形状 | 参数量 |
|------|------|--------|
| **标准投影矩阵** |  |  |
| W_Q | (512, 512) | 262,144 |
| W_K | (512, 512) | 262,144 |
| W_V | (512, 512) | 262,144 |
| W_O | (512, 512) | 262,144 |
| **相对位置嵌入** |  |  |
| Relative Pos Embeddings | (8, 65, 64) | 33,280 |
| **总计** |  | **1,081,856** |

### 关键观察

1. **相对位置嵌入数量**：$2 \times \text{max\_relative\_position} + 1 = 65$
2. **参数量与序列长度无关**：无论序列是128还是4096，相对位置嵌入参数量不变
3. **vs 绝对位置编码**：绝对位置需要为每个可能的位置存储嵌入

### 不同max_relative_position的参数量

| max_relative_position | 相对位置嵌入数 | 相对位置参数 | 总参数 |
|----------------------|---------------|-------------|--------|
| 16 | 33 | 16,896 | 1,065,472 |
| 32 | 65 | 33,280 | 1,081,856 |
| 64 | 129 | 66,048 | 1,114,624 |
| 128 | 257 | 131,584 | 1,180,160 |

**权衡**：
- 较小的max_relative_position：参数少，但远距离共享嵌入
- 较大的max_relative_position：更精细的远距离建模，但参数多
- T5通常使用32或128，已足够捕获大多数有用信息

## 相对位置 vs 绝对位置

### 对比表

| 特性 | 相对位置注意力 | 绝对位置编码 |
|------|---------------|-------------|
| **位置表示** | 相对距离 (i-j) | 绝对索引 (0, 1, 2, ...) |
| **长度泛化** | ✅ 优秀 | ❌ 较差 |
| **参数量** | 固定（与序列长度无关） | 随最大序列长度增长 |
| **位置不变性** | ✅ 是 | ❌ 否 |
| **训练复杂度** | 稍高（需要计算位置得分） | 简单 |
| **推理灵活性** | ✅ 支持任意长度 | ❌ 受训练长度限制 |
| **语言建模** | 更自然（相对距离） | 较不自然 |

### 长度泛化示例

**场景**：模型在序列长度128上训练

| 方法 | 训练长度 | 测试长度256 | 测试长度512 |
|------|---------|------------|------------|
| **相对位置** | ✅ | ✅ | ✅ |
| **绝对位置** | ✅ | ⚠️ 需要外推 | ❌ 性能下降 |

## 优势与局限

### 优势

1. **长度泛化能力强**
   - 可以处理任意长度的序列
   - 不受训练时序列长度限制
   - 适合处理可变长度输入

2. **参数效率高**
   - 使用max_relative_position限制参数量
   - 参数量与序列长度无关
   - 远距离位置共享嵌入

3. **更自然的语言建模**
   - 语言理解更依赖相对位置（"前一个词"）
   - 位置不变性
   - 捕获局部依赖关系

4. **实验效果好**
   - T5等模型证明有效性
   - 在多种NLP任务上表现优秀

### 局限

1. **计算复杂度稍高**
   - 需要额外计算相对位置得分
   - 实现比绝对位置稍复杂

2. **远距离建模受限**
   - 超过max_relative_position的距离共享嵌入
   - 可能损失部分远距离信息

3. **内存开销**
   - 需要存储相对位置嵌入
   - 多头注意力时每个头都有自己的嵌入

## 实际应用

### T5 (Text-To-Text Transfer Transformer)

T5全面使用相对位置注意力：

```python
# T5-Base配置
embed_dim = 768
num_heads = 12
max_relative_position = 32  # 标准配置

# T5-Large
embed_dim = 1024
num_heads = 16
max_relative_position = 32
```

**T5的设计选择**：
- max_relative_position=32或128（取决于版本）
- 对编码器和解码器都使用相对位置注意力
- 双向注意力：可以看到前后的相对位置
- 超过max的距离共享相同嵌入

### Transformer-XL

Transformer-XL使用相对位置编码处理长序列：

```python
# Transformer-XL配置
embed_dim = 512
num_heads = 8
max_relative_position = 512  # 支持更长的相对距离
```

**特点**：
- 段级循环机制 + 相对位置编码
- 可以处理超长序列（数千个token）
- 避免绝对位置带来的长度限制

### DeBERTa

DeBERTa使用解耦的内容和位置注意力：

```python
# DeBERTa的改进
# 分别计算内容到内容、内容到位置、位置到内容的注意力
score = content_to_content + content_to_position + position_to_content
```

**创新点**：
- 解耦的注意力机制
- 相对位置编码的改进版本
- 在多个任务上超越BERT

### 使用示例

```python
# 创建相对位置注意力层
rpa = RelativePositionAttention(
    embed_dim=512,
    num_heads=8,
    max_relative_position=32
)

# 前向传播
x = np.random.randn(seq_len, 512)
output = rpa.forward(x)

print(f"输入: {x.shape}")    # (seq_len, 512)
print(f"输出: {output.shape}")  # (seq_len, 512)
```

### 自回归生成（GPT风格）

```python
# 带因果mask的相对位置注意力
causal_mask = np.tril(np.ones((seq_len, seq_len)))
output = rpa.forward(x, mask=causal_mask)
```

## 如何选择max_relative_position

### 经验法则

| 场景 | 建议配置 | 原因 |
|------|---------|------|
| **短文本** | 16-32 | 参数少，足够覆盖常见距离 |
| **中等文本** | 32-64 | 平衡参数量和建模能力 |
| **长文本** | 64-128 | 更好的远距离建模 |
| **超长文本** | 128-256 | Transformer-XL等场景 |

### 实验建议

1. **从32开始**：这是T5等模型验证过的配置
2. **观察注意力分布**：分析超过max距离的注意力权重
3. **任务相关调整**：
   - 局部任务（如NER）：较小的max即可
   - 全局任务（如文档分类）：可能需要较大的max

### 代码示例

```python
# 不同任务的配置
configs = {
    'ner': {'max_relative_position': 16},          # 命名实体识别
    'sentiment': {'max_relative_position': 32},     # 情感分析
    'summarization': {'max_relative_position': 64}, # 摘要生成
    'qa': {'max_relative_position': 128},           # 问答系统
}
```

## 复杂度分析

假设序列长度为$n$，嵌入维度为$d$，注意力头数为$h$：

| 操作 | 复杂度 |
|------|--------|
| Q, K, V投影 | $O(n \cdot d^2)$ |
| 内容得分计算 | $O(n^2 \cdot d)$ |
| 相对位置得分计算 | $O(n^2 \cdot d)$ |
| Softmax | $O(n^2)$ |
| 加权求和 | $O(n^2 \cdot d)$ |
| 输出投影 | $O(n \cdot d^2)$ |
| **总计** | **$O(n^2 \cdot d + n \cdot d^2)$** |

**与标准注意力对比**：
- 时间复杂度相同（主导项都是$O(n^2 \cdot d)$）
- 空间复杂度略高（需要存储相对位置嵌入）
- 实际运行时间稍慢（额外的位置得分计算）

## 可视化理解

### 相对位置的影响

```
Query位置5关注不同Key位置：

相对距离    Key位置    注意力权重    解释
-5          0         0.05         很远的过去
-3          2         0.12         较远的过去
-1          4         0.25         前一个位置 ⬅
 0          5         0.30         当前位置 ◉
+1          6         0.18         后一个位置 ➡
+3          8         0.08         较远的未来
+5          10        0.02         很远的未来
```

### 注意力模式可视化

```
不同相对距离的注意力权重分布：

0.30 │              ▓▓
     │            ▓▓▓▓▓▓
0.20 │          ▓▓▓▓▓▓▓▓▓▓
     │        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
0.10 │      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     │    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
0.00 └────────────────────────────
     -5  -3  -1   0  +1  +3  +5
        相对距离（负=过去，正=未来）

观察：注意力权重随相对距离增大而衰减
```

## 常见问题

### Q1: 相对位置注意力比绝对位置慢吗？
A: 稍慢，因为需要额外计算相对位置得分。但差异不大（通常<10%），长度泛化能力的提升值得这个代价。

### Q2: max_relative_position应该设多大？
A: 取决于任务。T5使用32，Transformer-XL使用512。一般32-128足够，更大的值参数增加但收益递减。

### Q3: 可以用于所有Transformer模型吗？
A: 是的。编码器、解码器、编码器-解码器架构都可以使用。T5就是全面使用相对位置注意力的例子。

### Q4: 如何处理超过max_relative_position的距离？
A: 超过max的距离会被裁剪（clip）到max，共享相同的相对位置嵌入。实验表明这不会显著影响性能。

### Q5: 相对位置编码和相对位置注意力有什么区别？
A:
- **相对位置编码**：在输入中加入相对位置信息（如Transformer-XL）
- **相对位置注意力**：在注意力计算中加入相对位置得分（如T5）
- 两者可以结合使用

### Q6: 训练时和推理时的序列长度可以不同吗？
A: 可以！这正是相对位置注意力的优势。只要max_relative_position覆盖了主要的相对距离，就可以处理任意长度。

## 实现技巧

### 1. 高效的相对位置得分计算

```python
# 方法1：循环（清晰但慢）
for i in range(seq_len):
    for j in range(seq_len):
        position_scores[i, j] = Q[i] @ relative_embeddings[rel_idx[i, j]]

# 方法2：向量化（快）
# 使用高级索引和einsum
position_scores = np.einsum('bhid,bijd->bhij', Q, relative_embeddings_gathered)
```

### 2. 缓存相对位置矩阵

```python
# 预计算常用长度的相对位置矩阵
class RelativePositionCache:
    def __init__(self, max_len, max_relative_position):
        self.cache = {}
        for length in [64, 128, 256, 512, 1024]:
            if length <= max_len:
                self.cache[length] = get_relative_positions(length, max_relative_position)
```

### 3. 共享相对位置嵌入

```python
# 在多个层之间共享相对位置嵌入
shared_rel_pos_emb = np.random.randn(num_heads, num_rel_pos, head_dim)

for layer in transformer_layers:
    layer.relative_position_embeddings = shared_rel_pos_emb  # 共享
```

## 参考文献

1. **原始论文**：Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). "Self-Attention with Relative Position Representations." NAACL.

2. **T5**：Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR.

3. **Transformer-XL**：Dai, Z., et al. (2019). "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." ACL.

4. **DeBERTa**：He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." ICLR.

5. **XLNet**：Yang, Z., et al. (2019). "XLNet: Generalized Autoregressive Pretraining for Language Understanding." NeurIPS.

## 文件说明

- `relative_position_attention.py`: Python实现（带详细中文注释）
- `relative_position_attention.ipynb`: Jupyter Notebook交互式演示
- `README.md`: 本说明文档

## 总结

Relative Position Attention通过编码相对位置而非绝对位置，提供了更好的长度泛化能力和更自然的语言建模方式。

**核心要点**：
- ✅ 相对距离编码，位置不变性
- ✅ 长度泛化能力强，不受训练长度限制
- ✅ 参数量与序列长度无关
- ✅ 被T5、Transformer-XL等先进模型采用
- ✅ 更自然的语言理解方式

**适用场景**：
- 需要处理可变长度输入的任务
- 长文本理解和生成
- 需要长度泛化的场景
- 局部依赖关系重要的任务

**设计建议**：
- max_relative_position通常选择32-128
- T5的配置（max=32）是良好的起点
- 根据任务特点调整max值
- 考虑参数量和建模能力的权衡

相对位置注意力是现代Transformer架构的重要改进，为处理长序列和提升模型泛化能力提供了有效方案！
