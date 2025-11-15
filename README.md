# 🚀 LLM算法白板编程 - 面试准备仓库

这是一个专注于大语言模型（LLM）领域经典算法的学习仓库，适合准备机器学习/深度学习面试的同学。每个算法都包含详细的中文注释、数学原理、代码实现和可视化演示。

## 📚 仓库特点

- ✅ **详细的中文注释**：每个算法都有完整的中文注释和文档
- ✅ **三种形式**：Python代码 + Jupyter Notebook + Markdown文档
- ✅ **从零实现**：所有算法都从基础开始手写实现
- ✅ **可视化演示**：包含丰富的图表和可视化
- ✅ **理论结合实践**：数学公式 + 代码实现 + 应用案例
- ✅ **面试友好**：覆盖常见的LLM面试算法题

## 📖 目录结构

### 01. 基础注意力机制

注意力机制是现代LLM的基础，理解这些基础概念对后续学习至关重要。

| 算法 | 说明 | 关键特点 | 应用 |
|------|------|---------|------|
| [Soft Attention](./01_基础注意力机制/soft_attention/) | 软注意力 | 对所有位置计算权重，可微分 | Seq2seq、图像描述 |
| [Hard Attention](./01_基础注意力机制/hard_attention/) | 硬注意力 | 只选择部分位置，不可微分 | 视觉任务、精确定位 |
| [Self-Attention](./01_基础注意力机制/self_attention/) | 自注意力 | 序列内部元素相互关注 | Transformer核心 |

### 02. Transformer家族

Transformer是现代LLM的基础架构，这部分包含其核心组件。

| 算法 | 说明 | 关键特点 | 应用 |
|------|------|---------|------|
| [Scaled Dot-Product Attention](./02_Transformer家族/scaled_dot_product_attention/) | 缩放点积注意力 | 通过sqrt(d_k)缩放防止梯度消失 | Transformer基础 |
| [Multi-Head Attention](./02_Transformer家族/multi_head_attention/) | 多头注意力 | 并行多个注意力头捕获不同模式 | BERT、GPT核心 |
| [Cross-Attention](./02_Transformer家族/cross_attention/) | 交叉注意力 | 连接编码器和解码器 | 机器翻译、T5 |

### 03. 高效注意力变体

为了解决标准注意力O(n²)复杂度问题，研究者提出了多种高效变体。

| 算法 | 说明 | 复杂度 | 关键技术 | 应用模型 |
|------|------|--------|---------|---------|
| [Linear Attention](./03_高效注意力变体/linear_attention/) | 线性注意力 | O(n) | Kernel trick | Performer、RWKV |
| [Sparse Attention](./03_高效注意力变体/sparse_attention/) | 稀疏注意力 | O(n√n) | 局部+全局模式 | Longformer、BigBird |
| [Flash Attention](./03_高效注意力变体/flash_attention/) | Flash注意力 | O(n²) 但快2-4x | GPU内存优化 | 几乎所有现代LLM |
| [Grouped Query Attention (GQA)](./03_高效注意力变体/grouped_query_attention/) | 分组查询注意力 | 减少25%参数 | 多Q头共享KV | LLaMA 2、Mistral |
| [Multi-Query Attention (MQA)](./03_高效注意力变体/multi_query_attention/) | 多查询注意力 | 减少43%参数 | 所有Q头共享KV | PaLM、Falcon |
| [MLA](./03_高效注意力变体/mla/) | Multi-Head Latent Attention | KV缓存减少75% | 低秩KV压缩 | DeepSeek-V2/V3 |

### 04. 其他注意力变体

这些变体在特定场景下有独特优势。

| 算法 | 说明 | 关键特点 | 应用 |
|------|------|---------|------|
| [Relative Position Attention](./04_其他注意力变体/relative_position_attention/) | 相对位置注意力 | 编码相对位置而非绝对位置 | T5、Transformer-XL |
| [Sliding Window Attention](./04_其他注意力变体/sliding_window_attention/) | 滑动窗口注意力 | 只关注固定窗口内的token | Mistral、Longformer |
| [Ring Attention](./04_其他注意力变体/ring_attention/) | 环形注意力 | 分布式计算长序列 | 超长上下文（100万token+） |

### 05. 核心组件

LLM的关键构建模块。

| 组件 | 说明 | 关键特点 | 应用 |
|------|------|---------|------|
| [Decoder](./05_核心组件/decoder/) | Transformer解码器 | Self-Attn + Cross-Attn + FFN | GPT、LLaMA、T5 |
| [Layer Norm](./05_核心组件/layer_norm/) | 层归一化 | 稳定训练，包含RMSNorm | 所有Transformer |
| [LoRA](./05_核心组件/lora/) | 低秩适应 | 参数高效微调（减少10000x） | 大模型微调标配 |
| [KV Cache](./05_核心组件/kv_cache/) | 键值缓存 | 加速自回归生成 | 所有LLM推理 |

### 06. 优化算法

训练神经网络的基础优化算法。

| 算法 | 说明 | 关键特点 | 应用 |
|------|------|---------|------|
| [Gradient Descent](./06_优化算法/gradient_descent/) | 梯度下降 | SGD、Momentum、Adam等 | 所有深度学习 |
| [Backpropagation](./06_优化算法/backpropagation/) | 反向传播 | 计算图和梯度流 | 神经网络训练基础 |

### 07. 强化学习算法（RLHF）

用于大模型对齐的关键算法。

| 算法 | 说明 | 关键特点 | 应用 |
|------|------|---------|------|
| [PPO](./07_强化学习算法/ppo/) | 近端策略优化 | RLHF标准算法 | ChatGPT、Claude |
| [DPO](./07_强化学习算法/dpo/) | 直接偏好优化 | 无需奖励模型 | Zephyr、Mistral |
| [GRPO](./07_强化学习算法/grpo/) | 组相对策略优化 | Group内相对比较 | DeepSeek |
| [DAPO](./07_强化学习算法/dapo/) | 分布式优势策略优化 | 考虑优势分布 | 风险敏感决策 |
| [GSPO](./07_强化学习算法/gspo/) | 组随机策略优化 | 随机采样优化 | 样本效率优化 |

### 08. 机器学习基础

基础但重要的机器学习算法。

| 算法 | 说明 | 关键特点 | 应用 |
|------|------|---------|------|
| [K-Means](./08_机器学习基础/kmeans/) | K均值聚类 | 无监督聚类、肘部法则 | 数据分析、特征提取 |
| [MSE](./08_机器学习基础/mse/) | 均方误差 | 回归损失函数 | 线性回归、神经网络 |

## 🎯 快速开始

### 环境要求

```bash
python >= 3.8
numpy >= 1.20.0
matplotlib >= 3.3.0
jupyter >= 1.0.0
seaborn >= 0.11.0
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

#### 方式1: 运行Python脚本

```bash
# 运行Soft Attention示例
python 01_基础注意力机制/soft_attention/soft_attention.py

# 运行Multi-Head Attention示例
python 02_Transformer家族/multi_head_attention/multi_head_attention.py

# 运行DPO示例
python 07_强化学习算法/dpo/dpo.py
```

#### 方式2: Jupyter Notebook

```bash
# 启动Jupyter Notebook
jupyter notebook

# 然后打开任意.ipynb文件，例如：
# 01_基础注意力机制/self_attention/self_attention.ipynb
```

## 📝 学习路径建议

### 初级路径（适合入门）

1. **基础概念**
   - [Self-Attention](./01_基础注意力机制/self_attention/) - 理解注意力机制的核心
   - [Scaled Dot-Product Attention](./02_Transformer家族/scaled_dot_product_attention/) - Transformer的基础

2. **核心组件**
   - [Multi-Head Attention](./02_Transformer家族/multi_head_attention/) - 多头机制
   - [Layer Norm](./05_核心组件/layer_norm/) - 归一化层
   - [Decoder](./05_核心组件/decoder/) - 完整的Decoder结构

3. **优化基础**
   - [Gradient Descent](./06_优化算法/gradient_descent/) - 优化算法
   - [Backpropagation](./06_优化算法/backpropagation/) - 反向传播

### 中级路径（准备面试）

1. **高效注意力**
   - [GQA](./03_高效注意力变体/grouped_query_attention/) - 理解KV共享
   - [MQA](./03_高效注意力变体/multi_query_attention/) - 极致优化
   - [Flash Attention](./03_高效注意力变体/flash_attention/) - GPU优化

2. **实用组件**
   - [KV Cache](./05_核心组件/kv_cache/) - 推理加速
   - [LoRA](./05_核心组件/lora/) - 参数高效微调

3. **对齐算法**
   - [DPO](./07_强化学习算法/dpo/) - 直接优化
   - [PPO](./07_强化学习算法/ppo/) - 强化学习

### 高级路径（深入研究）

1. **前沿技术**
   - [MLA](./03_高效注意力变体/mla/) - DeepSeek创新
   - [Ring Attention](./04_其他注意力变体/ring_attention/) - 超长上下文
   - [Linear Attention](./03_高效注意力变体/linear_attention/) - O(n)复杂度

2. **完整系统**
   - 组合各个组件构建完整的LLM
   - 实现端到端的训练和推理

## 💡 使用建议

### 学习每个算法的步骤

1. **阅读README.md**
   - 理解算法的核心思想
   - 了解数学原理和应用场景

2. **查看Python实现**
   - 阅读详细的中文注释
   - 理解代码实现细节

3. **运行Jupyter Notebook**
   - 交互式体验算法
   - 调整参数观察效果

4. **手写实现**
   - 合上资料，尝试自己实现
   - 对比你的代码和仓库中的实现

5. **深入理解**
   - 思考算法的优缺点
   - 考虑在实际项目中如何应用

## 🎓 面试准备建议

### 常见面试算法（必须掌握）

1. ⭐⭐⭐ **Self-Attention** - 几乎必考
2. ⭐⭐⭐ **Multi-Head Attention** - 高频考点
3. ⭐⭐⭐ **Layer Norm** - 常见手撕题
4. ⭐⭐ **KV Cache** - 推理优化考点
5. ⭐⭐ **LoRA** - 微调相关面试
6. ⭐⭐ **DPO** - 对齐算法热门
7. ⭐ **GQA/MQA** - 效率优化考点

### 面试技巧

1. **先说思路**：讲清楚算法的核心思想
2. **画图辅助**：注意力矩阵、架构图等
3. **分析复杂度**：时间、空间复杂度
4. **举实际例子**：哪些模型用了这个算法
5. **讨论优缺点**：什么场景适合/不适合

### 手撕代码准备

建议重点练习以下几个：
- Self-Attention的完整实现
- Multi-Head Attention（包含Q、K、V投影）
- Layer Norm（包含反向传播）
- Softmax（数值稳定版本）
- KV Cache的使用
- LoRA的实现

## 📊 复杂度对比

| 算法 | 时间复杂度 | 空间复杂度 | KV缓存 | 适用场景 |
|------|-----------|-----------|--------|---------|
| Standard Attention | O(n²d) | O(n²) | 100% | 通用 |
| Multi-Head Attention | O(n²d) | O(n²) | 100% | 通用 |
| Linear Attention | O(nd²) | O(d²) | - | 超长序列 |
| Sparse Attention | O(n√n·d) | O(n√n) | - | 长文档 |
| Flash Attention | O(n²d) | O(n) | 100% | 所有场景（标配） |
| GQA (8头→2KV) | O(n²d) | O(n²) | 25% | 平衡选择 |
| MQA (8头→1KV) | O(n²d) | O(n²) | 12.5% | 效率优先 |

## 🌟 现代LLM架构图

```
输入Token
    ↓
Embedding + Position Encoding
    ↓
┌─────────────────────┐
│  Decoder Layers ×N  │
│  ┌────────────────┐ │
│  │ Self-Attention │ │  ← 使用GQA/MQA优化
│  │   (Causal)     │ │  ← Flash Attention加速
│  ├────────────────┤ │
│  │  Layer Norm    │ │  ← RMSNorm变体
│  ├────────────────┤ │
│  │      FFN       │ │
│  ├────────────────┤ │
│  │  Layer Norm    │ │
│  └────────────────┘ │
└─────────────────────┘
    ↓
Output Projection
    ↓
生成Token (使用KV Cache加速)
    ↓
RLHF对齐 (DPO/PPO)
```

## 🔧 代码规范

本仓库所有代码遵循以下规范：

- **语言**：Python 3.8+
- **依赖**：仅使用NumPy（便于理解底层实现）
- **注释**：详细的中文注释
- **文档**：每个算法都有完整的README
- **示例**：可运行的示例代码
- **测试**：包含基本的正确性验证

## 📚 参考资料

### 经典论文

1. **Attention Is All You Need** (2017) - Transformer原始论文
2. **BERT** (2018) - 双向编码器
3. **GPT-2/3** (2019/2020) - 自回归生成
4. **Flash Attention** (2022) - GPU优化
5. **LLaMA** (2023) - 开源大模型
6. **DPO** (2023) - 直接偏好优化

### 推荐资源

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Andrej Karpathy的神经网络系列](https://karpathy.ai/)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

如果你想贡献新的算法实现，请遵循以下格式：
1. 创建对应的目录
2. 实现`.py`文件（带详细中文注释）
3. 创建`.ipynb` Notebook演示
4. 编写`README.md`文档
5. 更新主README的目录

## 📄 许可证

MIT License

## ⭐ Star History

如果这个仓库对你有帮助，请给个Star支持一下！

---

**祝你面试顺利！加油！💪**

如有问题，欢迎提Issue讨论。
