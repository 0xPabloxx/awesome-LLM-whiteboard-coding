# LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，通过在预训练模型的权重矩阵旁添加低秩分解矩阵来适配下游任务，大幅减少可训练参数量。

## 📖 核心概念

### 基本原理

传统微调需要更新所有参数，而LoRA通过低秩矩阵适配：

```
h = W₀x + ΔWx = W₀x + BAx
```

其中：
- `W₀ ∈ R^(d×k)`: 冻结的预训练权重
- `B ∈ R^(d×r)`: 可训练的低秩矩阵B
- `A ∈ R^(r×k)`: 可训练的低秩矩阵A
- `r << min(d, k)`: 秩，决定参数量

### 参数量对比

**全量微调**: `d × k` 参数
**LoRA**: `r × (d + k)` 参数

当 `r << d, k` 时，参数量大幅减少。

**示例** (d=k=512, r=8):
- 全量微调: 262,144 参数
- LoRA: 8,192 参数
- 压缩比: **32x**
- 参数减少: **96.9%**

## 🎯 核心优势

### 1. 参数高效
- ✅ 通常减少 **99%+** 可训练参数
- ✅ 在消费级GPU上微调大模型
- ✅ 显存需求大幅降低

### 2. 推理无开销
- ✅ 可合并权重: `W_merged = W₀ + α/r · BA`
- ✅ 推理速度与原模型完全相同
- ✅ 无额外计算或显存开销

### 3. 多任务灵活
- ✅ 共享预训练权重
- ✅ 每任务独立LoRA模块
- ✅ 快速切换任务

### 4. 保持泛化能力
- ✅ 预训练权重冻结
- ✅ 保留原模型知识
- ✅ 避免灾难性遗忘

## 🔧 实现细节

### 基础LoRA层

```python
class LoRALayer:
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        # 冻结的预训练权重
        self.W = pretrained_weight  # (out_features, in_features)

        # 可训练的低秩矩阵
        self.A = np.random.randn(rank, in_features) / np.sqrt(in_features)
        self.B = np.zeros((out_features, rank))  # 初始化为0

        # 缩放因子
        self.scaling = alpha / rank

    def forward(self, x):
        # 原始分支（冻结）
        output = np.dot(x, self.W.T)

        # LoRA分支（可训练）
        lora_output = np.dot(np.dot(x, self.A.T), self.B.T)

        # 合并
        return output + lora_output * self.scaling

    def merge_weights(self):
        """推理优化：合并权重"""
        return self.W + self.scaling * np.dot(self.B, self.A)
```

### 权重初始化

```python
# A: 使用Kaiming初始化
A = np.random.randn(rank, in_features) / np.sqrt(in_features)

# B: 初始化为0（确保初始时ΔW=0）
B = np.zeros((out_features, rank))
```

### 缩放因子

```python
scaling = alpha / rank
```

通常设置 `alpha=16` 或 `alpha=32`，这样当改变rank时，LoRA的贡献度保持相对稳定。

## 📊 参数量分析

### 不同Rank的对比

以 `d=k=512` 为例：

| Rank | LoRA参数 | 占比 | 压缩比 |
|------|---------|------|--------|
| 1 | 1,024 | 0.39% | 256.0x |
| 2 | 2,048 | 0.78% | 128.0x |
| 4 | 4,096 | 1.56% | 64.0x |
| 8 | 8,192 | 3.12% | 32.0x |
| 16 | 16,384 | 6.25% | 16.0x |
| 32 | 32,768 | 12.50% | 8.0x |
| 64 | 65,536 | 25.00% | 4.0x |

### BERT-base规模

配置：`hidden_dim=768, num_layers=12`

**场景1: Attention层（仅Q/V投影，rank=8）**

```
全量微调: 12层 × 4投影 × 768² = 28,311,552 参数
LoRA: 12层 × 2投影 × (8×768 + 768×8) = 147,456 参数

参数减少: 99.48%
压缩比: 192x
```

**场景2: 完整Transformer（Attention+FFN）**

```
全量微调: 约85M参数
LoRA (仅Attention Q/V): 约147K参数

压缩比: 576x
```

## 🎨 在多头注意力中应用

### 标准配置

通常在以下投影中应用LoRA：

```python
class MultiHeadAttentionWithLoRA:
    def __init__(self, embed_dim, num_heads, rank=8):
        # 常见配置：只在Q和V上使用LoRA
        self.W_q = LoRALayer(embed_dim, embed_dim, rank)
        self.W_k = pretrained_wk  # 冻结，不用LoRA
        self.W_v = LoRALayer(embed_dim, embed_dim, rank)
        self.W_o = pretrained_wo  # 冻结，不用LoRA
```

### 不同配置的参数量

以 `embed_dim=768, rank=8` 为例：

| 配置 | 可训练参数 | 占比 |
|------|-----------|------|
| 无LoRA（全量） | 2,359,296 | 100% |
| 仅Q | 12,288 | 0.52% |
| **Q和V（常用）** | **24,576** | **1.04%** |
| Q,K,V | 36,864 | 1.56% |
| 全部(Q,K,V,O) | 49,152 | 2.08% |

## 🔬 Rank选择指南

### 推荐配置

| Rank | 适用场景 | 参数效率 | 效果 |
|------|---------|---------|------|
| 1-4 | 极简任务、极度资源受限 | 极高 | 基础 |
| **8** | **通用推荐、大多数任务** | **高** | **良好** |
| 16 | 中等复杂任务 | 中等 | 较好 |
| 32-64 | 复杂任务、领域差异大 | 较低 | 接近全量 |

### 实验建议

```python
# 从rank=8开始实验
lora = LoRALayer(d_in, d_out, rank=8, alpha=16)

# 如果效果不理想，尝试增大rank
ranks_to_try = [8, 16, 32]  # 逐步增大
```

## 🏗️ 实际应用案例

### 1. GPT-3 微调

```
模型: GPT-3 (175B参数)
任务: 领域适配

全量微调: 175B参数
LoRA (rank=4): ~37.7M参数

参数减少: 99.98%
显存需求: 从1.2TB降至350GB
```

### 2. LLaMA微调（Alpaca）

```
模型: LLaMA-7B
任务: 指令跟随

配置:
- 应用位置: Attention的Q/V投影
- Rank: 8
- Alpha: 16

结果:
- 可训练参数: 4.2M
- 训练时间: 3小时 (8×A100)
- 效果: 接近全量微调
```

### 3. Stable Diffusion风格迁移

```
模型: Stable Diffusion 1.5
任务: 艺术风格定制

配置:
- 应用位置: Cross-Attention层
- Rank: 4-8
- 文件大小: 3-10MB

优势:
- 快速训练（几分钟到几小时）
- 轻量模型（便于分享）
- 可组合多个LoRA
```

### 4. 多任务学习

```python
class MultiTaskSystem:
    def __init__(self, base_model):
        # 共享的预训练权重
        self.base_model = base_model  # 冻结

        # 为每个任务创建LoRA模块
        self.task_loras = {
            'sentiment': LoRAModule(rank=8),
            'ner': LoRAModule(rank=8),
            'qa': LoRAModule(rank=16),
            'summarization': LoRAModule(rank=16)
        }

    def infer(self, x, task):
        # 加载对应任务的LoRA
        return self.base_model(x) + self.task_loras[task](x)
```

## 💡 最佳实践

### 1. 应用位置选择

```python
# 推荐：只在Attention的Q和V投影上应用
use_lora_on = ['q', 'v']  # 最常用

# 其他选择：
# ['q', 'k', 'v']  # 所有Attention投影
# ['q', 'v', 'o']  # 包括输出投影
# ['q', 'v'] + FFN  # 扩展到FFN层
```

### 2. 超参数设置

```python
# 标准配置
rank = 8           # 大多数任务的起点
alpha = 16         # 通常为rank的2倍
dropout = 0.05     # 可选，防止过拟合

# 初始化
A ~ N(0, 1/√k)     # Kaiming初始化
B = 0              # 零初始化
```

### 3. 训练技巧

```python
# 1. 冻结预训练权重
for param in base_model.parameters():
    param.requires_grad = False

# 2. 只训练LoRA参数
trainable_params = [lora.A, lora.B for lora in lora_layers]

# 3. 使用较大的学习率
optimizer = Adam(trainable_params, lr=1e-4)  # 比全量微调大10-100倍

# 4. 较少的训练步数
# LoRA通常收敛更快
```

### 4. 推理优化

```python
# 训练后合并权重
W_merged = W_0 + (alpha / rank) * B @ A

# 替换原始权重
model.weight = W_merged

# 删除LoRA模块
del lora_layers

# 现在推理速度与原模型相同
```

## 📈 性能对比

### 显存占用

以LLaMA-7B为例（batch_size=8, seq_len=512）：

| 方法 | 显存占用 | 相对节省 |
|------|---------|---------|
| 全量微调 | ~60GB | 基准 |
| LoRA (rank=8) | ~14GB | 77% |
| LoRA (rank=4) | ~12GB | 80% |

### 训练速度

```
相同硬件配置下：
全量微调: 100%（基准）
LoRA: 120-150%（更快！）

原因：
- 更少的参数需要梯度
- 更少的优化器状态
- 更好的显存利用
```

## ⚠️ 局限性

### 1. 效果权衡
- 在某些任务上可能略逊于全量微调
- 需要实验找到合适的rank

### 2. 适用范围
- 主要用于线性层
- 卷积层需要特殊处理
- 不适合改变模型架构

### 3. 领域适配
- 对于领域差异极大的任务，可能需要更大的rank或全量微调

## 🔗 扩展阅读

### 论文
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)

### 实现库
- **HuggingFace PEFT**: 官方LoRA实现
- **llama.cpp**: LoRA推理支持
- **stable-diffusion-webui**: Stable Diffusion的LoRA

### 变体
- **QLoRA**: LoRA + 量化，进一步减少显存
- **AdaLoRA**: 自适应调整每层的rank
- **LoRA-FA**: 使用固定的A矩阵

## 🚀 快速开始

```bash
# 运行Python脚本
python lora.py

# 运行Jupyter notebook
jupyter notebook lora.ipynb
```

### 简单示例

```python
import numpy as np

# 创建LoRA层
lora = LoRALayer(
    in_features=512,
    out_features=512,
    rank=8,
    alpha=16
)

# 前向传播
x = np.random.randn(4, 512)
output = lora.forward(x)

# 合并权重用于推理
W_merged = lora.merge_weights()
```

## 📊 应用统计

### 现代LLM微调方案

| 模型 | 微调方法 | Rank | 说明 |
|------|---------|------|------|
| Alpaca | LoRA | 8 | LLaMA-7B指令微调 |
| Vicuna | LoRA | 8 | LLaMA-13B对话微调 |
| ChatGLM | LoRA | 8 | 中文对话模型 |
| Guanaco | QLoRA | 64 | 4-bit量化+LoRA |

### 参数效率对比

```
方法              可训练参数    显存      训练时间
─────────────────────────────────────────────
全量微调          100%          100%      100%
BitFit           0.1%          80%       85%
Adapter          2-4%          85%       90%
Prefix-Tuning    0.1-1%        75%       80%
LoRA (rank=8)    0.1-1%        25%       60%
QLoRA (rank=64)  2-3%          15%       70%
```

---

**关键要点**：LoRA通过低秩分解实现参数高效微调，在保持模型性能的同时大幅减少可训练参数和显存需求，是当前大模型微调的标准方案之一。
