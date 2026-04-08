# LLM 白板编程算法地图

> 28 个 LLM 领域经典算法，每个算法包含 Python 实现 + Jupyter Notebook 交互演示 + README 文档

---

## 01. 基础注意力机制

| 算法 | 说明 | 目录 |
|------|------|------|
| Soft Attention | 软注意力，基于加权求和的注意力机制 | `01_基础注意力机制/soft_attention/` |
| Hard Attention | 硬注意力，基于采样的离散注意力机制 | `01_基础注意力机制/hard_attention/` |
| Self-Attention | 自注意力，序列内部元素之间的注意力计算 | `01_基础注意力机制/self_attention/` |

## 02. Transformer 家族

| 算法 | 说明 | 目录 |
|------|------|------|
| Scaled Dot-Product Attention | 缩放点积注意力，Transformer 的核心计算单元 | `02_Transformer家族/scaled_dot_product_attention/` |
| Multi-Head Attention | 多头注意力，并行多组注意力捕获不同子空间信息 | `02_Transformer家族/multi_head_attention/` |
| Cross-Attention | 交叉注意力，用于 encoder-decoder 之间的信息交互 | `02_Transformer家族/cross_attention/` |

## 03. 高效注意力变体

| 算法 | 说明 | 目录 |
|------|------|------|
| Linear Attention | 线性注意力，将复杂度从 O(n²) 降到 O(n) | `03_高效注意力变体/linear_attention/` |
| Sparse Attention | 稀疏注意力，只计算部分位置的注意力以提升效率 | `03_高效注意力变体/sparse_attention/` |
| Flash Attention | GPU 显存优化的精确注意力算法（tiling + recomputation） | `03_高效注意力变体/flash_attention/` |
| Grouped Query Attention (GQA) | 分组查询注意力，多个 query head 共享一组 KV head | `03_高效注意力变体/grouped_query_attention/` |
| Multi-Query Attention (MQA) | 多查询注意力，所有 query head 共享同一组 KV | `03_高效注意力变体/multi_query_attention/` |
| Multi-Head Latent Attention (MLA) | DeepSeek 提出的低秩压缩 KV 的高效注意力 | `03_高效注意力变体/mla/` |

## 04. 其他注意力变体

| 算法 | 说明 | 目录 |
|------|------|------|
| Relative Position Attention | 相对位置注意力，编码 token 间相对距离而非绝对位置 | `04_其他注意力变体/relative_position_attention/` |
| Sliding Window Attention | 滑动窗口注意力，限制注意力范围以处理长序列 | `04_其他注意力变体/sliding_window_attention/` |
| Ring Attention | 环形注意力，跨设备分布式计算超长序列注意力 | `04_其他注意力变体/ring_attention/` |

## 05. 核心组件

| 算法 | 说明 | 目录 |
|------|------|------|
| Decoder | Transformer 解码器，自回归生成的核心模块 | `05_核心组件/decoder/` |
| Layer Norm | 层归一化（含 RMSNorm），稳定训练的关键组件 | `05_核心组件/layer_norm/` |
| LoRA | 低秩适应，参数高效微调方法 | `05_核心组件/lora/` |
| KV Cache | 键值缓存，加速自回归推理的核心优化 | `05_核心组件/kv_cache/` |

## 06. 优化算法

| 算法 | 说明 | 目录 |
|------|------|------|
| Gradient Descent | 梯度下降（SGD / Momentum / Adam） | `06_优化算法/gradient_descent/` |
| Backpropagation | 反向传播，计算梯度的链式法则 | `06_优化算法/backpropagation/` |

## 07. 强化学习算法（RLHF）

| 算法 | 说明 | 目录 |
|------|------|------|
| PPO | Proximal Policy Optimization，RLHF 的主流训练算法 | `07_强化学习算法/ppo/` |
| DPO | Direct Preference Optimization，无需 reward model 的对齐方法 | `07_强化学习算法/dpo/` |
| GRPO | Group Relative Policy Optimization，DeepSeek 提出的组相对策略优化 | `07_强化学习算法/grpo/` |
| DAPO | Dynamic sampling And Partial Ordering，动态采样偏好优化 | `07_强化学习算法/dapo/` |
| GSPO | Group Selective Policy Optimization，组选择策略优化 | `07_强化学习算法/gspo/` |

## 08. 机器学习基础

| 算法 | 说明 | 目录 |
|------|------|------|
| K-Means | K 均值聚类，经典无监督学习算法 | `08_机器学习基础/kmeans/` |
| MSE | 均方误差，回归任务的基础损失函数 | `08_机器学习基础/mse/` |

---

## 快速统计

- **总算法数**: 28
- **总文件数**: 87（每个算法 3 件套：`.py` + `.ipynb` + `README.md`）
- **覆盖领域**: 注意力机制 → Transformer → 高效推理 → 训练优化 → RLHF 对齐 → ML 基础
