"""
Linear Attention (线性注意力) 实现

线性注意力通过kernel trick将注意力的复杂度从O(n²)降低到O(n)。
标准注意力需要计算所有位置对的相似度，而线性注意力通过改变计算顺序避免了显式的注意力矩阵。

核心思想：
1. 标准注意力: Softmax(QK^T)V = D^(-1)exp(QK^T)V
2. 线性注意力: φ(Q)(φ(K)^TV) 其中φ是特征映射函数
3. 先计算(φ(K)^TV)可以将复杂度从O(n²d)降到O(nd²)

优势：
- 计算复杂度O(n) vs 标准注意力O(n²)
- 内存占用O(n) vs 标准注意力O(n²)
- 适合处理长序列

劣势：
- 性能通常略低于标准注意力
- 不能使用softmax（改用其他激活函数）

应用：Performer、Linear Transformer等模型
"""

import numpy as np
import time


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def elu_feature_map(x):
    """
    ELU+1特征映射函数

    这是一个简单但有效的特征映射，保证输出非负
    φ(x) = elu(x) + 1 = max(0, x) + min(0, exp(x) - 1) + 1

    Args:
        x: 输入张量

    Returns:
        非负特征映射
    """
    return np.maximum(0, x) + np.minimum(0, np.exp(x) - 1) + 1


def random_fourier_features(x, num_features=None, scale=1.0, seed=42):
    """
    随机傅里叶特征（RFF）

    使用随机投影来近似RBF核，这是Performer论文中使用的方法
    φ(x) = sqrt(2/m) * [cos(ωx + b), sin(ωx + b)]

    Args:
        x: 输入张量，形状 (n, d)
        num_features: 特征数量，默认为2*d
        scale: 随机投影的标准差
        seed: 随机种子

    Returns:
        特征映射，形状 (n, num_features)
    """
    n, d = x.shape
    if num_features is None:
        num_features = 2 * d

    np.random.seed(seed)
    # 随机投影矩阵
    omega = np.random.randn(d, num_features // 2) * scale
    # 随机偏置
    b = np.random.uniform(0, 2 * np.pi, num_features // 2)

    # 计算投影
    projection = np.dot(x, omega) + b

    # 正余弦特征
    features = np.concatenate([
        np.cos(projection),
        np.sin(projection)
    ], axis=-1)

    return features * np.sqrt(2.0 / num_features)


class StandardAttention:
    """
    标准注意力机制（用于对比）

    复杂度: O(n²d)
    内存: O(n²)
    """

    def __init__(self, embed_dim):
        """
        初始化标准注意力层

        Args:
            embed_dim: 嵌入维度
        """
        self.embed_dim = embed_dim

        # 初始化Q、K、V的投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, x, return_time=False):
        """
        前向传播

        Args:
            x: 输入序列，形状 (seq_len, embed_dim)
            return_time: 是否返回计算时间

        Returns:
            output: 输出序列，形状 (seq_len, embed_dim)
            (可选) time: 计算时间
        """
        start_time = time.time()

        seq_len, embed_dim = x.shape

        # 线性投影得到Q、K、V
        Q = np.dot(x, self.W_q)  # (seq_len, embed_dim)
        K = np.dot(x, self.W_k)  # (seq_len, embed_dim)
        V = np.dot(x, self.W_v)  # (seq_len, embed_dim)

        # 计算注意力得分矩阵 (seq_len, seq_len)
        # 这一步的复杂度是O(n²d)
        scores = np.dot(Q, K.T) / np.sqrt(embed_dim)

        # Softmax归一化
        attention_weights = softmax(scores, axis=-1)

        # 加权求和 (seq_len, seq_len) @ (seq_len, embed_dim) = (seq_len, embed_dim)
        # 这一步的复杂度是O(n²d)
        output = np.dot(attention_weights, V)

        elapsed_time = time.time() - start_time

        if return_time:
            return output, elapsed_time
        return output


class LinearAttention:
    """
    线性注意力机制

    通过kernel trick改变计算顺序，避免显式计算注意力矩阵
    复杂度: O(nd²)，当d << n时约为O(n)
    内存: O(d²)
    """

    def __init__(self, embed_dim, feature_map='elu', num_features=None):
        """
        初始化线性注意力层

        Args:
            embed_dim: 嵌入维度
            feature_map: 特征映射类型 ('elu' 或 'rff')
            num_features: RFF特征数量（仅当feature_map='rff'时使用）
        """
        self.embed_dim = embed_dim
        self.feature_map_type = feature_map
        self.num_features = num_features if num_features else embed_dim

        # 初始化Q、K、V的投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def feature_map(self, x):
        """应用特征映射函数"""
        if self.feature_map_type == 'elu':
            return elu_feature_map(x)
        elif self.feature_map_type == 'rff':
            return random_fourier_features(x, self.num_features)
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map_type}")

    def forward(self, x, return_time=False):
        """
        前向传播（线性复杂度）

        核心技巧：改变计算顺序
        标准: Attention(Q,K,V) = Softmax(QK^T)V
        线性: Attention(Q,K,V) = φ(Q)(φ(K)^TV) / (φ(Q)sum(φ(K)))

        通过先计算φ(K)^TV（形状为d×d），避免了n×n的注意力矩阵

        Args:
            x: 输入序列，形状 (seq_len, embed_dim)
            return_time: 是否返回计算时间

        Returns:
            output: 输出序列，形状 (seq_len, embed_dim)
            (可选) time: 计算时间
        """
        start_time = time.time()

        seq_len, embed_dim = x.shape

        # 线性投影得到Q、K、V
        Q = np.dot(x, self.W_q)  # (seq_len, embed_dim)
        K = np.dot(x, self.W_k)  # (seq_len, embed_dim)
        V = np.dot(x, self.W_v)  # (seq_len, embed_dim)

        # 应用特征映射
        Q_prime = self.feature_map(Q)  # (seq_len, feature_dim)
        K_prime = self.feature_map(K)  # (seq_len, feature_dim)

        # 关键步骤：先计算 K^T @ V
        # 这避免了计算完整的注意力矩阵
        # (feature_dim, seq_len) @ (seq_len, embed_dim) = (feature_dim, embed_dim)
        KV = np.dot(K_prime.T, V)  # 复杂度 O(n * feature_dim * embed_dim)

        # 计算归一化项：每个query位置的分母
        # sum(φ(K), axis=0) 得到 (feature_dim,)
        K_sum = np.sum(K_prime, axis=0, keepdims=True)  # (1, feature_dim)

        # 对每个query位置：
        # numerator = φ(Q) @ (φ(K)^T @ V)
        # denominator = φ(Q) @ sum(φ(K))
        numerator = np.dot(Q_prime, KV)  # (seq_len, embed_dim)
        denominator = np.dot(Q_prime, K_sum.T)  # (seq_len, 1)

        # 归一化
        output = numerator / (denominator + 1e-6)

        elapsed_time = time.time() - start_time

        if return_time:
            return output, elapsed_time
        return output


def compare_attention_mechanisms(seq_lengths, embed_dim=64):
    """
    比较标准注意力和线性注意力的性能

    Args:
        seq_lengths: 要测试的序列长度列表
        embed_dim: 嵌入维度
    """
    print("=" * 80)
    print("标准注意力 vs 线性注意力 - 性能对比")
    print("=" * 80)
    print(f"\n嵌入维度: {embed_dim}")
    print(f"{'序列长度':<12} {'标准注意力(ms)':<18} {'线性注意力(ms)':<18} {'加速比':<12}")
    print("-" * 80)

    for seq_len in seq_lengths:
        # 生成随机输入
        x = np.random.randn(seq_len, embed_dim)

        # 标准注意力
        std_attn = StandardAttention(embed_dim)
        _, std_time = std_attn.forward(x, return_time=True)

        # 线性注意力 (ELU)
        linear_attn = LinearAttention(embed_dim, feature_map='elu')
        _, linear_time = linear_attn.forward(x, return_time=True)

        speedup = std_time / linear_time

        print(f"{seq_len:<12} {std_time*1000:<18.4f} {linear_time*1000:<18.4f} {speedup:<12.2f}x")

    print("-" * 80)


def visualize_feature_maps(embed_dim=8):
    """
    可视化不同特征映射函数的效果

    Args:
        embed_dim: 嵌入维度
    """
    print("\n" + "=" * 80)
    print("特征映射函数对比")
    print("=" * 80)

    # 生成测试向量
    x = np.random.randn(5, embed_dim)

    print("\n原始向量 (前3行):")
    print(x[:3])

    # ELU特征映射
    print("\nELU+1 特征映射 (前3行):")
    elu_features = elu_feature_map(x)
    print(elu_features[:3])
    print(f"范围: [{elu_features.min():.4f}, {elu_features.max():.4f}]")

    # RFF特征映射
    print("\n随机傅里叶特征 (前3行，前8列):")
    rff_features = random_fourier_features(x, num_features=16)
    print(rff_features[:3, :8])
    print(f"形状: {rff_features.shape}")
    print(f"范围: [{rff_features.min():.4f}, {rff_features.max():.4f}]")


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("线性注意力机制演示")
    print("=" * 80)

    # 参数设置
    seq_len = 64
    embed_dim = 32

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 1. 标准注意力
    print("\n1. 标准注意力")
    print("-" * 80)
    std_attn = StandardAttention(embed_dim)
    output_std, time_std = std_attn.forward(x, return_time=True)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_std.shape}")
    print(f"计算时间: {time_std*1000:.4f} ms")
    print(f"复杂度: O(n²d) = O({seq_len}² × {embed_dim}) = O({seq_len**2 * embed_dim})")

    # 2. 线性注意力 (ELU)
    print("\n2. 线性注意力 (ELU特征映射)")
    print("-" * 80)
    linear_attn_elu = LinearAttention(embed_dim, feature_map='elu')
    output_linear_elu, time_linear = linear_attn_elu.forward(x, return_time=True)
    print(f"输出形状: {output_linear_elu.shape}")
    print(f"计算时间: {time_linear*1000:.4f} ms")
    print(f"加速比: {time_std/time_linear:.2f}x")
    print(f"复杂度: O(nd²) = O({seq_len} × {embed_dim}²) = O({seq_len * embed_dim**2})")

    # 3. 线性注意力 (RFF)
    print("\n3. 线性注意力 (随机傅里叶特征)")
    print("-" * 80)
    num_features = embed_dim * 2
    linear_attn_rff = LinearAttention(embed_dim, feature_map='rff', num_features=num_features)
    output_linear_rff, time_rff = linear_attn_rff.forward(x, return_time=True)
    print(f"输出形状: {output_linear_rff.shape}")
    print(f"特征数量: {num_features}")
    print(f"计算时间: {time_rff*1000:.4f} ms")
    print(f"加速比: {time_std/time_rff:.2f}x")

    # 4. 输出差异分析
    print("\n4. 输出差异分析")
    print("-" * 80)
    diff_elu = np.mean(np.abs(output_std - output_linear_elu))
    diff_rff = np.mean(np.abs(output_std - output_linear_rff))
    print(f"标准注意力 vs 线性注意力(ELU) - 平均绝对差异: {diff_elu:.6f}")
    print(f"标准注意力 vs 线性注意力(RFF) - 平均绝对差异: {diff_rff:.6f}")
    print("注: 线性注意力是近似，输出会有差异")

    # 5. 性能对比
    print("\n5. 不同序列长度下的性能对比")
    compare_attention_mechanisms([32, 64, 128, 256, 512], embed_dim=32)

    # 6. 特征映射可视化
    visualize_feature_maps(embed_dim=8)

    print("\n" + "=" * 80)
    print("线性注意力的关键优势:")
    print("=" * 80)
    print("✓ 复杂度从O(n²)降低到O(n) (当d << n时)")
    print("✓ 内存占用从O(n²)降低到O(d²)")
    print("✓ 适合处理超长序列")
    print("✓ 保持了注意力机制的核心能力")
    print("\n权衡:")
    print("✗ 性能通常略低于标准注意力")
    print("✗ 需要选择合适的特征映射函数")
    print("✗ 不能直接使用softmax")
    print("=" * 80)
