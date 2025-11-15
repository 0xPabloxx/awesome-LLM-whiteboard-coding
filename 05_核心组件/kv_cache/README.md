# KV Cache: 键值缓存

KV Cache是自回归生成中的关键优化技术，通过缓存已计算的Key和Value来避免重复计算，大幅提升推理速度。

## 📖 核心概念

### 为什么需要KV Cache？

在自回归生成中，每生成一个新token都需要关注之前的所有token：

```
Step 1: "The"          → 计算 K₁, V₁
Step 2: "The cat"      → 计算 K₁, V₁, K₂, V₂  ← K₁,V₁重复计算！
Step 3: "The cat sat"  → 计算 K₁, V₁, K₂, V₂, K₃, V₃  ← 更多重复！
```

**问题：** 随着序列变长，重复计算量呈平方级增长。

**KV Cache解决方案：**
```
Step 1: 计算并缓存 K₁, V₁
Step 2: 只计算 K₂, V₂，从缓存读取 K₁, V₁
Step 3: 只计算 K₃, V₃，从缓存读取 K₁, V₁, K₂, V₂
```

## 🎯 核心优势

### 1. 性能提升
- ✅ 推理速度提升 **2-3倍**
- ✅ 长序列提升更明显
- ✅ 计算量从 O(n²) 降到 O(n)

### 2. 计算优化
**无缓存（每步）：**
```
计算量 = n × d + n² × d (投影 + 注意力)
```

**有缓存（每步）：**
```
计算量 = 1 × d + n × d (只投影新token + 注意力)
```

### 3. 显存换速度
- 需要额外显存存储K、V
- 显存开销随序列长度线性增长
- 推理速度提升值得这个开销

## 🔧 实现细节

### KVCache类

```python
class KVCache:
    def __init__(self, num_heads, head_dim, max_seq_len=2048):
        # 预分配缓存空间
        self.k_cache = np.zeros((num_heads, max_seq_len, head_dim))
        self.v_cache = np.zeros((num_heads, max_seq_len, head_dim))
        self.cache_len = 0

    def update(self, new_k, new_v):
        """添加新的K、V到缓存"""
        new_seq_len = new_k.shape[1]

        # 添加到缓存
        self.k_cache[:, self.cache_len:self.cache_len+new_seq_len, :] = new_k
        self.v_cache[:, self.cache_len:self.cache_len+new_seq_len, :] = new_v

        self.cache_len += new_seq_len

        # 返回完整的K、V
        return (self.k_cache[:, :self.cache_len, :],
                self.v_cache[:, :self.cache_len, :])
```

### 两阶段生成流程

#### 阶段1: Prefill（预填充）

处理输入prompt，计算所有token的K、V：

```python
# 输入：完整prompt
prompt = tokens[0:prompt_len]  # 例如: "The cat sat on"

# 计算所有token的Q, K, V
Q = project_q(prompt)  # (prompt_len, d)
K = project_k(prompt)  # (prompt_len, d)
V = project_v(prompt)  # (prompt_len, d)

# 初始化缓存
kv_cache.update(K, V)

# 计算attention（可并行）
output = attention(Q, K, V)
```

**特点：**
- 计算密集型
- 可以并行处理所有token
- 一次性填充缓存

#### 阶段2: Decode（解码）

逐个生成新token，使用缓存：

```python
for step in range(max_new_tokens):
    # 只处理最新的token
    new_token = generated[step]

    # 只计算新token的Q, K, V
    Q_new = project_q(new_token)  # (1, d)
    K_new = project_k(new_token)  # (1, d)
    V_new = project_v(new_token)  # (1, d)

    # 更新缓存
    K_full, V_full = kv_cache.update(K_new, V_new)

    # 使用完整缓存计算attention
    output = attention(Q_new, K_full, V_full)

    # 采样下一个token
    next_token = sample(output)
```

**特点：**
- 显存带宽密集型
- 必须序列化处理
- 增量更新缓存

## 📊 性能分析

### 计算量对比

| 序列长度 | 无缓存（每步） | 有缓存（每步） | 加速比 |
|---------|--------------|--------------|--------|
| 10 | 100 | 11 | 9.1x |
| 50 | 2,500 | 51 | 49.0x |
| 100 | 10,000 | 101 | 99.0x |
| 500 | 250,000 | 501 | 499.0x |
| 1000 | 1,000,000 | 1,001 | 999.0x |

**结论：** 序列越长，加速越明显！

### 显存占用

对于LLaMA-7B (32层，32头，头维度128，最大长度2048)：

```python
# 每层KV Cache大小
per_layer = 2 × num_heads × seq_len × head_dim × sizeof(float16)
         = 2 × 32 × 2048 × 128 × 2 bytes
         = 32 MB

# 总KV Cache（32层）
total = 32 × 32 MB = 1 GB (batch_size=1)
```

**不同batch size的显存占用：**

| Batch Size | KV Cache显存 |
|-----------|-------------|
| 1 | 1.0 GB |
| 4 | 4.0 GB |
| 8 | 8.0 GB |
| 16 | 16.0 GB |
| 32 | 32.0 GB |

### 时间分布

假设prompt长度50，生成100个token：

```
Prefill: 50² = 2,500 ops (约40%时间)
Decode:  Σ(50+i) for i in 1..100 = 10,000 ops (约60%时间)
```

**特点：**
- Prefill是一次性开销
- Decode时间随生成长度线性增长
- 生成越长，Decode占比越大

## 🚀 优化技术

### 1. Multi-Query Attention (MQA)

所有头共享同一个K、V：

```python
# 标准MHA
K: (num_heads, seq_len, head_dim)  # 例如: (32, 2048, 128)
V: (num_heads, seq_len, head_dim)

# MQA
K: (1, seq_len, head_dim)  # 只有1组！
V: (1, seq_len, head_dim)
```

**优势：**
- 显存减少 `num_heads` 倍
- LLaMA-7B: 从1GB降到32MB
- 速度提升（更少的显存访问）

**劣势：**
- 效果可能略有下降

**应用：** PaLM、Falcon

### 2. Grouped-Query Attention (GQA)

多个头共享一组K、V，平衡效果和显存：

```python
# GQA-8 (32头分8组，每组4头)
num_kv_heads = num_heads // 4 = 8

K: (8, seq_len, head_dim)
V: (8, seq_len, head_dim)
```

**优势：**
- 显存减少4倍（相比MHA）
- 效果接近MHA（优于MQA）
- 灵活调节分组数

**应用：** LLaMA-2、Mistral

### 3. 量化KV Cache

使用低精度存储K、V：

```python
# FP16 → INT8
K_int8 = quantize(K_fp16)
V_int8 = quantize(V_fp16)

# 使用时反量化
K_fp16 = dequantize(K_int8)
```

**优势：**
- 显存减少2倍（FP16→INT8）
- 显存减少4倍（FP16→INT4）
- 精度损失很小

### 4. Paged Attention (vLLM)

按块（page）管理KV缓存，类似操作系统的虚拟内存：

```python
# 将KV cache分成固定大小的块（例如64 tokens）
page_size = 64
num_pages = (seq_len + page_size - 1) // page_size

# 按需分配和释放页
pages = allocate_pages(num_pages)
```

**优势：**
- 提高显存利用率（减少碎片）
- 支持动态batch
- 高效处理变长序列

**应用：** vLLM推理框架

### 优化效果对比

以LLaMA-7B为例（batch_size=1, seq_len=2048）：

| 方法 | KV Cache显存 | 相对节省 |
|------|-------------|---------|
| 标准MHA (FP32) | 4.0 GB | 基准 |
| 标准MHA (FP16) | 2.0 GB | 50% |
| MQA (FP16) | 0.06 GB | 98.5% |
| GQA-8 (FP16) | 0.5 GB | 87.5% |
| 标准MHA (INT8) | 1.0 GB | 75% |
| MQA (INT8) | 0.03 GB | 99.25% |

## 🏗️ 实际应用

### GPT系列

```python
# GPT-3 (175B参数)
配置:
- 96层
- 96头
- 头维度: 128
- 最大长度: 2048

KV Cache (batch=1):
- 每层: 96MB
- 总计: 9.2GB
```

### LLaMA系列

```python
# LLaMA-7B
配置:
- 32层
- 32头（GQA: 8 KV头）
- 头维度: 128
- 最大长度: 4096

KV Cache (GQA, FP16, batch=1):
- 标准: 2GB
- GQA优化: 0.5GB
```

### 推理服务

**vLLM (Paged Attention):**
```
特点:
- 动态KV Cache管理
- 高吞吐量（比HuggingFace快24x）
- 支持大batch推理
```

**TensorRT-LLM:**
```
特点:
- INT8量化KV Cache
- 融合kernel优化
- 多GPU并行
```

## 💡 最佳实践

### 1. 选择合适的优化技术

```python
# 对于显存充足的场景
use_optimization = "Standard MHA + FP16"

# 对于显存受限的场景
use_optimization = "GQA + INT8"

# 对于极端显存受限
use_optimization = "MQA + INT8"
```

### 2. 预分配缓存空间

```python
# 好的做法：预分配
kv_cache = np.zeros((num_heads, max_seq_len, head_dim))

# 不好的做法：动态拼接
kv_cache = np.concatenate([kv_cache, new_kv], axis=1)  # 每次都重新分配！
```

### 3. Batch处理优化

```python
# Prefill阶段：尽可能大batch
prefill_batch_size = 32

# Decode阶段：平衡延迟和吞吐
decode_batch_size = 8  # 太大会增加延迟
```

### 4. 长序列处理

```python
# 对于超长序列，考虑滑动窗口
if seq_len > max_cache_len:
    # 只保留最近的N个token
    kv_cache = kv_cache[:, -max_cache_len:, :]
```

## 📈 性能调优

### 瓶颈分析

**Prefill阶段：**
- 瓶颈：计算（FLOPs）
- 优化：增大batch，使用Tensor Cores

**Decode阶段：**
- 瓶颈：显存带宽
- 优化：量化、MQA/GQA、Flash Attention

### 推理延迟估算

```python
def estimate_latency(prompt_len, num_generated, model_flops, memory_bandwidth):
    # Prefill延迟（计算受限）
    prefill_ops = prompt_len ** 2 * model_flops
    prefill_latency = prefill_ops / gpu_compute_power

    # Decode延迟（带宽受限）
    bytes_per_token = kv_cache_size * 2  # 读K、V
    decode_latency = (bytes_per_token * num_generated) / memory_bandwidth

    return prefill_latency + decode_latency
```

## ⚠️ 注意事项

### 1. 显存管理

```python
# 问题：OOM（显存不足）
solution_1 = "减小batch_size"
solution_2 = "使用GQA/MQA"
solution_3 = "量化KV Cache"
solution_4 = "减小max_seq_len"
```

### 2. 精度损失

```python
# INT8量化可能导致轻微精度损失
# 建议：先实验评估影响

# 对于关键任务，保持FP16
kv_cache_dtype = "fp16"

# 对于一般任务，INT8足够
kv_cache_dtype = "int8"
```

### 3. 并发请求

```python
# 问题：多个请求共享KV Cache会冲突

# 解决：每个请求独立缓存
kv_caches = {
    request_id: KVCache(...) for request_id in requests
}
```

## 🔗 相关资源

### 论文
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)

### 开源项目
- **vLLM**: 高性能LLM推理（Paged Attention）
- **TensorRT-LLM**: NVIDIA官方推理加速
- **Text Generation Inference**: HuggingFace推理服务

## 🚀 快速开始

```bash
# 运行Python脚本
python kv_cache.py

# 运行Jupyter notebook
jupyter notebook kv_cache.ipynb
```

### 简单示例

```python
import numpy as np

# 创建KV Cache
kv_cache = KVCache(num_heads=8, head_dim=64, max_seq_len=2048)

# Prefill阶段
prompt_k = np.random.randn(8, 20, 64)
prompt_v = np.random.randn(8, 20, 64)
full_k, full_v = kv_cache.update(prompt_k, prompt_v)
print(f"Prefill后缓存长度: {kv_cache.cache_len}")

# Decode阶段
for step in range(10):
    new_k = np.random.randn(8, 1, 64)
    new_v = np.random.randn(8, 1, 64)
    full_k, full_v = kv_cache.update(new_k, new_v)
    print(f"Step {step+1}: 缓存长度 = {kv_cache.cache_len}")
```

## 📊 对比总结

| 特性 | 无KV Cache | 有KV Cache |
|------|-----------|-----------|
| **每步计算量** | O(n²) | O(n) |
| **推理速度** | 基准 | 2-3x |
| **显存占用** | 低 | 中-高 |
| **实现复杂度** | 简单 | 中等 |
| **是否标准** | ❌ | ✅ (所有LLM) |

---

**关键要点**：KV Cache是自回归生成的标准优化，通过缓存已计算的K、V来避免重复计算，是所有现代LLM推理的必备技术。结合MQA/GQA、量化等优化，可以在保持性能的同时大幅减少显存占用。
