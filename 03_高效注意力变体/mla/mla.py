"""
MLA (Multi-Head Latent Attention) 实现

MLA是DeepSeek-V2引入的高效注意力机制，通过低秩压缩KV缓存显著降低内存占用。
核心思想是将高维的KV投影到低维潜在空间，减少KV缓存的存储需求。

核心思想：
1. 标准MHA：每个头独立的KV投影，KV缓存大小 = n_heads × d_head × seq_len
2. MLA：所有头共享低秩的潜在表示，KV缓存大小 = d_latent × seq_len
3. 压缩比：通常可以减少75-90%的KV缓存

关键技术：
1. 低秩KV压缩：K, V = compress(x) → expand_per_head
2. 潜在空间共享：所有注意力头共享相同的压缩表示
3. 解耦的RoPE：位置编码单独处理

优势：
- KV缓存减少5-10倍
- 推理速度更快（特别是长序列）
- 性能损失很小
- 适合大规模部署

应用：DeepSeek-V2、DeepSeek-V3等模型
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def apply_rotary_pos_emb(x, cos, sin):
    """
    应用旋转位置编码（RoPE）

    Args:
        x: 输入张量 (seq_len, dim)
        cos: cosine部分 (seq_len, dim)
        sin: sine部分 (seq_len, dim)

    Returns:
        应用RoPE后的张量
    """
    # 将x分成两半，应用旋转
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # 重组以应用旋转
    rotated = np.zeros_like(x)
    rotated[..., ::2] = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    rotated[..., 1::2] = x1 * cos[..., 1::2] + x2 * sin[..., 1::2]

    return rotated


class StandardMultiHeadAttention:
    """
    标准多头注意力（用于对比）

    KV缓存大小：n_heads × d_head × seq_len
    """

    def __init__(self, embed_dim, n_heads):
        """
        初始化标准MHA

        Args:
            embed_dim: 嵌入维度
            n_heads: 注意力头数
        """
        assert embed_dim % n_heads == 0, "embed_dim必须能被n_heads整除"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        # 每个头独立的QKV投影
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, x, return_cache_size=False):
        """
        前向传播

        Args:
            x: 输入 (seq_len, embed_dim)
            return_cache_size: 是否返回KV缓存大小

        Returns:
            output: 输出 (seq_len, embed_dim)
            cache_size: (可选) KV缓存大小（字节）
        """
        seq_len, embed_dim = x.shape

        # QKV投影
        Q = np.dot(x, self.W_q).reshape(seq_len, self.n_heads, self.d_head)
        K = np.dot(x, self.W_k).reshape(seq_len, self.n_heads, self.d_head)
        V = np.dot(x, self.W_v).reshape(seq_len, self.n_heads, self.d_head)

        # 转置为 (n_heads, seq_len, d_head)
        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # 计算注意力
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_head)
        attention = softmax(scores, axis=-1)

        # 加权求和
        out = np.matmul(attention, V)  # (n_heads, seq_len, d_head)

        # 合并多头
        out = out.transpose(1, 0, 2).reshape(seq_len, embed_dim)

        # 输出投影
        output = np.dot(out, self.W_o)

        if return_cache_size:
            # KV缓存大小：2 (K和V) × n_heads × seq_len × d_head × 4 (float32)
            cache_size = 2 * self.n_heads * seq_len * self.d_head * 4
            return output, cache_size

        return output


class MultiHeadLatentAttention:
    """
    多头潜在注意力（MLA）

    核心创新：
    1. 低秩KV压缩：将KV投影到低维潜在空间
    2. 共享潜在表示：所有头共享相同的压缩KV
    3. 解耦RoPE：位置编码单独处理

    KV缓存大小：d_latent × seq_len (d_latent << n_heads × d_head)
    """

    def __init__(self, embed_dim, n_heads, d_latent=None, use_rope=False):
        """
        初始化MLA

        Args:
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            d_latent: 潜在空间维度（默认为embed_dim的1/4）
            use_rope: 是否使用RoPE
        """
        assert embed_dim % n_heads == 0, "embed_dim必须能被n_heads整除"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads
        self.d_latent = d_latent if d_latent else embed_dim // 4  # 默认压缩4倍
        self.use_rope = use_rope

        # Q投影（每个头独立）
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # KV压缩：先投影到低维潜在空间
        self.W_k_compress = np.random.randn(embed_dim, self.d_latent) / np.sqrt(embed_dim)
        self.W_v_compress = np.random.randn(embed_dim, self.d_latent) / np.sqrt(embed_dim)

        # KV解压缩：从潜在空间扩展到每个头
        self.W_k_expand = np.random.randn(self.d_latent, embed_dim) / np.sqrt(self.d_latent)
        self.W_v_expand = np.random.randn(self.d_latent, embed_dim) / np.sqrt(self.d_latent)

        # RoPE相关（如果使用）
        if use_rope:
            self.W_q_rope = np.random.randn(embed_dim, self.d_head) / np.sqrt(embed_dim)
            self.W_k_rope = np.random.randn(self.d_latent, self.d_head) / np.sqrt(self.d_latent)

        # 输出投影
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def _create_rope_embeddings(self, seq_len):
        """创建RoPE位置编码"""
        position = np.arange(seq_len)[:, None]
        div_term = np.exp(np.arange(0, self.d_head, 2) * -(np.log(10000.0) / self.d_head))

        cos = np.cos(position * div_term)
        sin = np.sin(position * div_term)

        # 扩展到完整维度
        cos_full = np.zeros((seq_len, self.d_head))
        sin_full = np.zeros((seq_len, self.d_head))

        cos_full[:, ::2] = cos
        cos_full[:, 1::2] = cos
        sin_full[:, ::2] = sin
        sin_full[:, 1::2] = sin

        return cos_full, sin_full

    def forward(self, x, return_cache_size=False):
        """
        MLA前向传播

        Args:
            x: 输入 (seq_len, embed_dim)
            return_cache_size: 是否返回KV缓存大小

        Returns:
            output: 输出 (seq_len, embed_dim)
            cache_size: (可选) KV缓存大小（字节）
        """
        seq_len, embed_dim = x.shape

        # 步骤1：Q投影（标准方式）
        Q = np.dot(x, self.W_q).reshape(seq_len, self.n_heads, self.d_head)

        # 步骤2：KV压缩到潜在空间
        K_compressed = np.dot(x, self.W_k_compress)  # (seq_len, d_latent)
        V_compressed = np.dot(x, self.W_v_compress)  # (seq_len, d_latent)

        # 步骤3：从潜在空间扩展到每个头
        K = np.dot(K_compressed, self.W_k_expand).reshape(seq_len, self.n_heads, self.d_head)
        V = np.dot(V_compressed, self.W_v_expand).reshape(seq_len, self.n_heads, self.d_head)

        # 步骤4：应用RoPE（如果使用）
        if self.use_rope:
            cos, sin = self._create_rope_embeddings(seq_len)

            # Q的RoPE
            Q_rope = np.dot(x, self.W_q_rope)  # (seq_len, d_head)
            Q_rope = apply_rotary_pos_emb(Q_rope, cos, sin)
            Q_rope = Q_rope[:, None, :].repeat(self.n_heads, axis=1)  # 广播到所有头

            # K的RoPE
            K_rope = np.dot(K_compressed, self.W_k_rope)  # (seq_len, d_head)
            K_rope = apply_rotary_pos_emb(K_rope, cos, sin)
            K_rope = K_rope[:, None, :].repeat(self.n_heads, axis=1)

            # 合并RoPE
            Q = Q + Q_rope
            K = K + K_rope

        # 转置为 (n_heads, seq_len, d_head)
        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # 步骤5：计算注意力
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_head)
        attention = softmax(scores, axis=-1)

        # 加权求和
        out = np.matmul(attention, V)  # (n_heads, seq_len, d_head)

        # 合并多头
        out = out.transpose(1, 0, 2).reshape(seq_len, embed_dim)

        # 输出投影
        output = np.dot(out, self.W_o)

        if return_cache_size:
            # KV缓存大小：2 (K和V) × d_latent × seq_len × 4 (float32)
            # 注意：只需要缓存压缩后的K_compressed和V_compressed
            cache_size = 2 * self.d_latent * seq_len * 4
            return output, cache_size

        return output


def compare_attention_mechanisms(seq_len=512, embed_dim=512, n_heads=8):
    """
    比较标准MHA和MLA

    Args:
        seq_len: 序列长度
        embed_dim: 嵌入维度
        n_heads: 注意力头数
    """
    print("=" * 80)
    print("标准多头注意力 vs 多头潜在注意力（MLA）")
    print("=" * 80)
    print(f"序列长度: {seq_len}, 嵌入维度: {embed_dim}, 头数: {n_heads}\n")

    # 生成输入
    x = np.random.randn(seq_len, embed_dim)

    # 标准MHA
    print("1. 标准多头注意力")
    print("-" * 80)
    std_mha = StandardMultiHeadAttention(embed_dim, n_heads)
    output_std, cache_std = std_mha.forward(x, return_cache_size=True)

    print(f"输出形状: {output_std.shape}")
    print(f"KV缓存大小: {cache_std / (1024*1024):.4f} MB")
    print(f"每个头的维度: {std_mha.d_head}")

    # MLA (不同压缩率)
    compression_ratios = [2, 4, 8]

    for ratio in compression_ratios:
        d_latent = embed_dim // ratio

        print(f"\n2. MLA (压缩率={ratio}x, d_latent={d_latent})")
        print("-" * 80)

        mla = MultiHeadLatentAttention(embed_dim, n_heads, d_latent=d_latent)
        output_mla, cache_mla = mla.forward(x, return_cache_size=True)

        print(f"输出形状: {output_mla.shape}")
        print(f"KV缓存大小: {cache_mla / (1024*1024):.4f} MB")
        print(f"潜在空间维度: {d_latent}")
        print(f"缓存节省: {(1 - cache_mla / cache_std) * 100:.1f}%")

        # 输出差异
        diff = np.abs(output_std - output_mla)
        print(f"与标准MHA的差异: {np.mean(diff):.6f}")

    print("\n" + "=" * 80)


def analyze_cache_scaling(embed_dim=512, n_heads=8):
    """
    分析不同序列长度下的KV缓存大小

    Args:
        embed_dim: 嵌入维度
        n_heads: 注意力头数
    """
    print("\n" + "=" * 80)
    print("KV缓存大小随序列长度的变化")
    print("=" * 80)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    print(f"\n{'序列长度':<12} {'标准MHA(MB)':<15} {'MLA-4x(MB)':<15} {'MLA-8x(MB)':<15} {'节省比例':<12}")
    print("-" * 80)

    for seq_len in seq_lengths:
        x = np.random.randn(seq_len, embed_dim)

        # 标准MHA
        std_mha = StandardMultiHeadAttention(embed_dim, n_heads)
        _, cache_std = std_mha.forward(x, return_cache_size=True)

        # MLA 4x压缩
        mla_4x = MultiHeadLatentAttention(embed_dim, n_heads, d_latent=embed_dim//4)
        _, cache_mla_4x = mla_4x.forward(x, return_cache_size=True)

        # MLA 8x压缩
        mla_8x = MultiHeadLatentAttention(embed_dim, n_heads, d_latent=embed_dim//8)
        _, cache_mla_8x = mla_8x.forward(x, return_cache_size=True)

        saving = (1 - cache_mla_4x / cache_std) * 100

        print(f"{seq_len:<12} {cache_std/(1024*1024):<15.4f} {cache_mla_4x/(1024*1024):<15.4f} "
              f"{cache_mla_8x/(1024*1024):<15.4f} {saving:<12.1f}%")

    print("-" * 80)


def visualize_compression():
    """
    可视化KV压缩过程
    """
    print("\n" + "=" * 80)
    print("MLA的KV压缩可视化")
    print("=" * 80)

    embed_dim = 512
    n_heads = 8
    d_head = embed_dim // n_heads
    d_latent = embed_dim // 4

    print(f"\n参数设置:")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {n_heads}")
    print(f"  每个头的维度: {d_head}")
    print(f"  潜在空间维度: {d_latent}")

    print(f"\n标准MHA的KV结构:")
    print(f"  K: (seq_len, n_heads × d_head) = (seq_len, {n_heads} × {d_head}) = (seq_len, {embed_dim})")
    print(f"  V: (seq_len, n_heads × d_head) = (seq_len, {n_heads} × {d_head}) = (seq_len, {embed_dim})")
    print(f"  每个token的KV大小: 2 × {embed_dim} = {2 * embed_dim} 维")

    print(f"\nMLA的KV结构:")
    print(f"  K_compressed: (seq_len, d_latent) = (seq_len, {d_latent})")
    print(f"  V_compressed: (seq_len, d_latent) = (seq_len, {d_latent})")
    print(f"  每个token的KV大小: 2 × {d_latent} = {2 * d_latent} 维")

    compression_ratio = (2 * embed_dim) / (2 * d_latent)
    print(f"\n压缩比: {compression_ratio:.1f}x")
    print(f"内存节省: {(1 - 1/compression_ratio) * 100:.1f}%")


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("多头潜在注意力（MLA）演示")
    print("=" * 80)

    # 基本对比
    compare_attention_mechanisms(seq_len=512, embed_dim=512, n_heads=8)

    # 缓存大小分析
    analyze_cache_scaling(embed_dim=512, n_heads=8)

    # 压缩可视化
    visualize_compression()

    print("\n" + "=" * 80)
    print("MLA的核心优势:")
    print("=" * 80)
    print("✓ KV缓存减少5-10倍")
    print("✓ 推理速度更快（特别是长序列）")
    print("✓ 支持更长的上下文")
    print("✓ 性能损失很小（通过训练补偿）")
    print("✓ 特别适合大规模部署")
    print("\n核心技术:")
    print("- 低秩KV压缩（降维）")
    print("- 所有头共享潜在表示")
    print("- 解耦的位置编码（RoPE）")
    print("- 压缩→扩展的两阶段设计")
    print("\n应用:")
    print("- DeepSeek-V2: 使用MLA处理长上下文")
    print("- DeepSeek-V3: 进一步优化的MLA")
    print("=" * 80)
