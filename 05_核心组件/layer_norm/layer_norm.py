"""
Layer Normalization (层归一化) 实现

层归一化是Transformer中的关键组件，用于稳定训练和加速收敛。
与Batch Normalization不同，Layer Norm在特征维度上进行归一化，
不依赖于batch，因此更适合序列模型和小batch场景。

核心思想：
1. 对每个样本的所有特征计算均值和方差
2. 使用均值和方差对特征进行标准化
3. 通过可学习的缩放和平移参数恢复表达能力

公式：
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
其中 μ 和 σ² 是该层所有特征的均值和方差

应用：所有Transformer架构（BERT、GPT、T5等）
"""

import numpy as np


class LayerNorm:
    """
    层归一化实现

    特点：
    - 对每个样本独立归一化（不依赖batch）
    - 在特征维度上计算统计量
    - 训练和推理时行为一致（无需moving average）
    - 更适合RNN/Transformer等序列模型
    """

    def __init__(self, normalized_shape, eps=1e-5, use_bias=True):
        """
        初始化层归一化

        Args:
            normalized_shape: 需要归一化的维度大小（通常是embed_dim）
            eps: 防止除零的小常数
            use_bias: 是否使用偏置β
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.use_bias = use_bias

        # 可学习参数
        self.gamma = np.ones(normalized_shape)  # 缩放参数
        self.beta = np.zeros(normalized_shape) if use_bias else None  # 平移参数

        # 缓存中间结果（用于反向传播）
        self.cache = {}

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (..., normalized_shape)
               可以是 (seq_len, embed_dim) 或 (batch, seq_len, embed_dim)

        Returns:
            out: 归一化后的输出，形状与输入相同
        """
        # 在最后一个维度上计算均值和方差
        mean = np.mean(x, axis=-1, keepdims=True)  # (..., 1)
        var = np.var(x, axis=-1, keepdims=True)    # (..., 1)

        # 标准化
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        # 缩放和平移
        out = self.gamma * x_normalized
        if self.use_bias:
            out = out + self.beta

        # 缓存用于反向传播
        self.cache = {
            'x': x,
            'mean': mean,
            'var': var,
            'x_normalized': x_normalized
        }

        return out

    def backward(self, dout):
        """
        反向传播（简化版，仅用于演示）

        Args:
            dout: 输出梯度

        Returns:
            dx: 输入梯度
        """
        x = self.cache['x']
        mean = self.cache['mean']
        var = self.cache['var']
        x_normalized = self.cache['x_normalized']

        N = self.normalized_shape

        # 计算参数梯度
        dgamma = np.sum(dout * x_normalized, axis=tuple(range(dout.ndim - 1)), keepdims=False)
        if self.use_bias:
            dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)), keepdims=False)

        # 计算输入梯度
        dx_normalized = dout * self.gamma

        # 反向传播通过标准化
        std = np.sqrt(var + self.eps)
        dx = (1. / N) * (1. / std) * (
            N * dx_normalized - np.sum(dx_normalized, axis=-1, keepdims=True) -
            x_normalized * np.sum(dx_normalized * x_normalized, axis=-1, keepdims=True)
        )

        return dx


class RMSNorm:
    """
    Root Mean Square Normalization (RMS归一化)

    RMSNorm是LayerNorm的简化版本，由论文"Root Mean Square Layer Normalization"提出。
    它去掉了均值中心化步骤，只使用RMS进行归一化。

    优点：
    - 计算更简单，速度更快
    - 参数更少（无偏置β）
    - 在LLaMA、GPT-NeoX等模型中表现良好

    公式：
    RMSNorm(x) = γ * x / sqrt(mean(x²) + ε)
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """
        初始化RMS归一化

        Args:
            normalized_shape: 需要归一化的维度大小
            eps: 防止除零的小常数
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 只有缩放参数，无平移参数
        self.gamma = np.ones(normalized_shape)

        # 缓存
        self.cache = {}

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (..., normalized_shape)

        Returns:
            out: 归一化后的输出
        """
        # 计算RMS（均方根）
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

        # 归一化
        x_normalized = x / rms

        # 缩放
        out = self.gamma * x_normalized

        # 缓存
        self.cache = {
            'x': x,
            'rms': rms,
            'x_normalized': x_normalized
        }

        return out


class BatchNorm:
    """
    Batch Normalization (批归一化)

    用于对比实验。BatchNorm在batch维度上进行归一化，
    在CV领域很成功，但在NLP中不如LayerNorm。

    公式：
    BatchNorm(x) = γ * (x - μ_batch) / sqrt(σ²_batch + ε) + β
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        初始化批归一化

        Args:
            num_features: 特征数量
            eps: 防止除零的小常数
            momentum: moving average的动量
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # 移动平均统计量（用于推理）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch, ..., num_features)

        Returns:
            out: 归一化后的输出
        """
        if self.training:
            # 训练模式：在batch维度上计算统计量
            # 对于2D输入 (batch, features)
            # 对于3D输入 (batch, seq_len, features)，在batch和seq_len上计算
            axes = tuple(range(x.ndim - 1))

            mean = np.mean(x, axis=axes)
            var = np.var(x, axis=axes)

            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理模式：使用移动平均
            mean = self.running_mean
            var = self.running_var

        # 标准化
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        # 缩放和平移
        out = self.gamma * x_normalized + self.beta

        return out


def compare_normalizations(x, seq_model=True):
    """
    比较不同归一化方法的效果

    Args:
        x: 输入张量
        seq_model: 是否是序列模型（影响BatchNorm的行为）

    Returns:
        results: 包含各种归一化结果的字典
    """
    if x.ndim == 2:
        batch_size, features = x.shape
    else:  # 3D
        batch_size, seq_len, features = x.shape
        if not seq_model:
            x = x.reshape(-1, features)

    results = {}

    # Layer Normalization
    ln = LayerNorm(features)
    results['layer_norm'] = ln.forward(x)

    # RMS Normalization
    rms = RMSNorm(features)
    results['rms_norm'] = rms.forward(x)

    # Batch Normalization
    bn = BatchNorm(features)
    results['batch_norm'] = bn.forward(x)

    return results


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("层归一化(Layer Normalization)演示")
    print("=" * 70)

    # 参数设置
    batch_size = 4
    seq_len = 10
    embed_dim = 8

    print(f"\n配置:")
    print(f"  Batch大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  嵌入维度: {embed_dim}")

    # 生成示例输入（3D: batch, seq_len, embed_dim）
    x = np.random.randn(batch_size, seq_len, embed_dim) * 2 + 1
    print(f"\n输入形状: {x.shape}")
    print(f"输入统计:")
    print(f"  全局均值: {np.mean(x):.4f}")
    print(f"  全局方差: {np.var(x):.4f}")

    # ========== 1. Layer Normalization ==========
    print("\n" + "=" * 70)
    print("1. Layer Normalization")
    print("=" * 70)

    ln = LayerNorm(embed_dim)
    ln_output = ln.forward(x)

    print(f"\nLayer Norm输出形状: {ln_output.shape}")
    print(f"\n归一化后的统计（沿特征维度）:")

    # 检查每个位置的均值和方差
    for i in range(min(3, batch_size)):
        for j in range(min(3, seq_len)):
            pos_mean = np.mean(ln_output[i, j])
            pos_var = np.var(ln_output[i, j])
            print(f"  位置[{i},{j}] - 均值: {pos_mean:.6f}, 方差: {pos_var:.6f}")

    print(f"\n可学习参数:")
    print(f"  gamma形状: {ln.gamma.shape}")
    print(f"  beta形状: {ln.beta.shape if ln.beta is not None else 'None'}")

    # ========== 2. RMS Normalization ==========
    print("\n" + "=" * 70)
    print("2. RMS Normalization (RMSNorm)")
    print("=" * 70)

    rms = RMSNorm(embed_dim)
    rms_output = rms.forward(x)

    print(f"\nRMSNorm输出形状: {rms_output.shape}")
    print(f"\nRMSNorm特点:")
    print(f"  ✓ 无均值中心化（不减去均值）")
    print(f"  ✓ 只有缩放参数gamma，无偏置beta")
    print(f"  ✓ 计算更简单，速度更快")
    print(f"  ✓ 在LLaMA等大模型中使用")

    # ========== 3. Layer Norm vs RMS Norm 对比 ==========
    print("\n" + "=" * 70)
    print("3. Layer Norm vs RMS Norm 输出对比")
    print("=" * 70)

    print(f"\nLayer Norm输出（第一个样本的第一个token）:")
    print(ln_output[0, 0])

    print(f"\nRMS Norm输出（第一个样本的第一个token）:")
    print(rms_output[0, 0])

    print(f"\n差异:")
    diff = np.abs(ln_output - rms_output)
    print(f"  平均绝对差异: {np.mean(diff):.6f}")
    print(f"  最大绝对差异: {np.max(diff):.6f}")

    # ========== 4. 与Batch Norm对比 ==========
    print("\n" + "=" * 70)
    print("4. Layer Norm vs Batch Norm 对比")
    print("=" * 70)

    bn = BatchNorm(embed_dim)
    bn_output = bn.forward(x)

    print(f"\nBatch Norm输出形状: {bn_output.shape}")

    print(f"\n归一化维度对比:")
    print(f"  Layer Norm: 在特征维度上归一化（每个样本独立）")
    print(f"  Batch Norm: 在batch维度上归一化（依赖batch统计量）")

    print(f"\nBatch Norm统计量（在batch和seq维度上）:")
    print(f"  running_mean: {bn.running_mean[:4]}")
    print(f"  running_var: {bn.running_var[:4]}")

    # ========== 5. 为什么Transformer用Layer Norm？ ==========
    print("\n" + "=" * 70)
    print("5. 为什么Transformer用Layer Norm而非Batch Norm？")
    print("=" * 70)

    print("\nLayer Norm的优势:")
    print("  ✓ 不依赖batch大小，训练和推理行为一致")
    print("  ✓ 对序列长度变化不敏感")
    print("  ✓ 适合小batch训练")
    print("  ✓ 在RNN/Transformer中效果更好")

    print("\nBatch Norm的劣势（在序列模型中）:")
    print("  ✗ 依赖batch统计量，batch太小会不稳定")
    print("  ✗ 训练和推理行为不一致（需要moving average）")
    print("  ✗ 序列长度变化会影响统计量")
    print("  ✗ 在NLP任务中表现不如Layer Norm")

    # ========== 6. 计算效率分析 ==========
    print("\n" + "=" * 70)
    print("6. 计算效率分析")
    print("=" * 70)

    import time

    # 创建大一点的输入
    large_x = np.random.randn(32, 512, 512)

    # Layer Norm
    ln_large = LayerNorm(512)
    start = time.time()
    for _ in range(10):
        _ = ln_large.forward(large_x)
    ln_time = (time.time() - start) / 10

    # RMS Norm
    rms_large = RMSNorm(512)
    start = time.time()
    for _ in range(10):
        _ = rms_large.forward(large_x)
    rms_time = (time.time() - start) / 10

    print(f"\n输入形状: (32, 512, 512)")
    print(f"Layer Norm平均时间: {ln_time*1000:.2f}ms")
    print(f"RMS Norm平均时间: {rms_time*1000:.2f}ms")
    print(f"加速比: {ln_time/rms_time:.2f}x")

    # ========== 7. 参数量对比 ==========
    print("\n" + "=" * 70)
    print("7. 参数量对比")
    print("=" * 70)

    feature_dim = 768  # BERT-base的hidden size

    print(f"\n对于特征维度 = {feature_dim}:")
    print(f"  Layer Norm参数: gamma({feature_dim}) + beta({feature_dim}) = {feature_dim * 2}")
    print(f"  RMS Norm参数: gamma({feature_dim}) = {feature_dim}")
    print(f"  Batch Norm参数: gamma({feature_dim}) + beta({feature_dim}) = {feature_dim * 2}")
    print(f"             + running_mean({feature_dim}) + running_var({feature_dim})")

    # ========== 8. 实际应用场景 ==========
    print("\n" + "=" * 70)
    print("8. 实际应用场景")
    print("=" * 70)

    print("\nLayer Norm:")
    print("  • BERT: 每个Transformer层后")
    print("  • GPT: 每个Transformer层前/后（Pre-LN / Post-LN）")
    print("  • T5: 使用RMSNorm变体")

    print("\nRMS Norm:")
    print("  • LLaMA: 所有Transformer层")
    print("  • GPT-NeoX: 所有Transformer层")
    print("  • 其他现代大模型")

    print("\nBatch Norm:")
    print("  • CNN模型（ResNet、VGG等）")
    print("  • 一般不用于Transformer")

    # ========== 9. Pre-LN vs Post-LN ==========
    print("\n" + "=" * 70)
    print("9. Pre-LN vs Post-LN")
    print("=" * 70)

    print("\nPost-LN (BERT风格):")
    print("  x = x + Sublayer(x)")
    print("  x = LayerNorm(x)")

    print("\nPre-LN (GPT风格):")
    print("  x = x + Sublayer(LayerNorm(x))")

    print("\nPre-LN的优势:")
    print("  ✓ 训练更稳定，尤其是深层网络")
    print("  ✓ 不需要learning rate warmup")
    print("  ✓ 梯度流动更好")

    print("\n" + "=" * 70)
    print("Layer Normalization的关键特性:")
    print("=" * 70)
    print("✓ Transformer的标准组件，稳定训练")
    print("✓ 不依赖batch，适合序列模型")
    print("✓ RMSNorm是简化版本，速度更快")
    print("✓ 比Batch Norm更适合NLP任务")
    print("✓ 广泛应用于所有现代LLM")
