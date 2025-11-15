# Layer Normalization (层归一化)

层归一化是Transformer中的关键组件，用于稳定训练和加速收敛。与Batch Normalization不同，Layer Norm在特征维度上进行归一化，不依赖于batch，因此更适合序列模型。

## 📖 核心概念

### 基本公式

**Layer Normalization:**
```
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
```

其中：
- `μ` = 该层所有特征的均值
- `σ²` = 该层所有特征的方差
- `γ` 和 `β` = 可学习的缩放和平移参数
- `ε` = 防止除零的小常数（通常为1e-5）

**RMS Normalization (简化版本):**
```
RMSNorm(x) = γ * x / sqrt(mean(x²) + ε)
```

## 🎯 核心特点

### Layer Norm
- ✅ **不依赖batch**: 每个样本独立归一化
- ✅ **训练推理一致**: 无需moving average
- ✅ **适合序列模型**: 在NLP中效果优于Batch Norm
- ✅ **小batch友好**: batch=1也能正常工作

### RMS Norm
- ✅ **更简单**: 去掉均值中心化
- ✅ **更快**: 减少计算量
- ✅ **更少参数**: 只有γ，无β
- ✅ **大模型首选**: LLaMA、GPT-NeoX等使用

## 🔄 与Batch Norm对比

| 特性 | Layer Norm | Batch Norm |
|------|-----------|-----------|
| **归一化维度** | 特征维度（横向） | Batch维度（纵向） |
| **依赖batch** | ❌ 不依赖 | ✅ 依赖 |
| **训练/推理** | 行为一致 | 行为不同 |
| **适用场景** | NLP、序列模型 | CV、CNN |
| **小batch** | ✅ 友好 | ❌ 不稳定 |

### 可视化对比

```
输入矩阵 (batch × features):
┌─────────────────┐
│ 1  2  3  4  5  │  ← Layer Norm在每行上归一化
│ 6  7  8  9  10 │
│ 11 12 13 14 15 │
└─────────────────┘
  ↑  ↑  ↑  ↑  ↑
  Batch Norm在每列上归一化
```

## 📊 实现细节

### 1. Layer Norm实现

```python
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = np.ones(normalized_shape)  # 缩放
        self.beta = np.zeros(normalized_shape)   # 平移
        self.eps = eps

    def forward(self, x):
        # 计算均值和方差（在特征维度）
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # 标准化
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # 缩放和平移
        return self.gamma * x_norm + self.beta
```

### 2. RMS Norm实现

```python
class RMSNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = np.ones(normalized_shape)
        self.eps = eps

    def forward(self, x):
        # 计算RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

        # 归一化并缩放
        return self.gamma * (x / rms)
```

## 🏗️ Transformer中的应用

### Post-LN (BERT风格)

```python
# 先计算，后归一化
x = x + Attention(x)
x = LayerNorm(x)

x = x + FFN(x)
x = LayerNorm(x)
```

### Pre-LN (GPT风格)

```python
# 先归一化，后计算
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Pre-LN优势:**
- ✅ 训练更稳定
- ✅ 不需要warmup
- ✅ 梯度流动更好
- ✅ 更适合深层网络

## 📈 性能对比

### 计算复杂度

对于输入 `(seq_len, hidden_dim)`:

| 方法 | 时间复杂度 | 参数量 |
|------|-----------|--------|
| Layer Norm | O(seq_len × hidden_dim) | 2 × hidden_dim |
| RMS Norm | O(seq_len × hidden_dim) | 1 × hidden_dim |
| Batch Norm | O(batch × seq_len × hidden_dim) | 2 × hidden_dim + running stats |

### 速度测试（相对于Layer Norm）

```
输入: (32, 512, 512)
Layer Norm:  1.00x (基准)
RMS Norm:    0.85x (快15%)
Batch Norm:  1.05x
```

## 🎓 实际应用案例

### 现代LLM中的使用

| 模型 | 归一化方式 | 位置 |
|------|-----------|------|
| **BERT** | Layer Norm | Post-LN |
| **GPT-2/3** | Layer Norm | Pre-LN |
| **LLaMA** | RMS Norm | Pre-LN |
| **GPT-NeoX** | RMS Norm | Pre-LN |
| **T5** | RMS Norm | Pre-LN |
| **PaLM** | RMS Norm | Pre-LN |

### 参数示例（BERT-base）

```python
# hidden_size = 768
ln = LayerNorm(768)

参数量:
- gamma: 768
- beta:  768
总计:   1,536 参数/层

BERT-base有24层 (12 encoder × 2 LN/layer)
总Layer Norm参数: 1,536 × 24 = 36,864
```

## 🔬 关键洞察

### 为什么Transformer用Layer Norm？

1. **序列长度变化**
   - Layer Norm不受序列长度影响
   - Batch Norm在不同长度序列上表现不稳定

2. **小Batch训练**
   - LLM训练常用小batch（显存限制）
   - Layer Norm在batch=1时也能正常工作

3. **训练推理一致性**
   - Layer Norm无需维护running statistics
   - 简化模型部署

4. **梯度流动**
   - Pre-LN改善深层网络的梯度传播
   - 减少梯度消失问题

### RMS Norm为何流行？

1. **效率**：减少约15%计算量
2. **简洁**：移除不必要的均值中心化
3. **效果**：大模型中效果与Layer Norm相当
4. **参数**：减少一半参数（无beta）

## 📝 使用建议

### 选择指南

- **Transformer模型**: 首选Layer Norm或RMS Norm
- **大规模LLM**: 推荐RMS Norm（效率更高）
- **CNN模型**: 使用Batch Norm
- **小batch训练**: 必须用Layer Norm/RMS Norm

### 最佳实践

```python
# 1. 使用Pre-LN（更稳定）
class TransformerBlock:
    def forward(self, x):
        # 推荐：Pre-LN
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# 2. 选择合适的epsilon
ln = LayerNorm(dim, eps=1e-5)  # PyTorch默认
ln = LayerNorm(dim, eps=1e-6)  # 混合精度训练

# 3. RMS Norm用于大模型
rms = RMSNorm(dim)  # LLaMA风格
```

## 🚀 运行示例

```bash
# 运行Python脚本
python layer_norm.py

# 运行Jupyter notebook
jupyter notebook layer_norm.ipynb
```

## 📚 参考资源

### 论文
- [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)

### 实现参考
- PyTorch: `torch.nn.LayerNorm`
- TensorFlow: `tf.keras.layers.LayerNormalization`

## 🔗 相关内容

- **Multi-Head Attention**: Layer Norm的主要应用场景
- **Transformer Decoder**: 包含多个Layer Norm层
- **优化算法**: Layer Norm影响梯度分布

---

**关键要点**：Layer Norm是Transformer的基础组件，通过在特征维度归一化，提供了比Batch Norm更适合序列模型的解决方案。RMS Norm作为简化版本，在现代大模型中越来越流行。
