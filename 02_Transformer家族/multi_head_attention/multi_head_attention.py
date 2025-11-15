"""
Multi-Head Attention (多头注意力) 实现

多头注意力是Transformer的核心组件，通过多个注意力头并行计算，
让模型能够同时关注不同子空间的信息，提升表达能力。

核心思想：
1. 将Q、K、V分别投影到h个不同的子空间（每个头一个）
2. 在每个子空间独立计算scaled dot-product attention
3. 将所有头的输出concatenate
4. 通过输出投影矩阵进行最终变换

公式：
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

应用：所有Transformer架构的基础（BERT、GPT、T5等）
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention

    Args:
        Q: Query矩阵，形状为 (seq_len, d_k)
        K: Key矩阵，形状为 (seq_len, d_k)
        V: Value矩阵，形状为 (seq_len, d_v)
        mask: 注意力掩码，形状为 (seq_len, seq_len)

    Returns:
        output: 输出，形状为 (seq_len, d_v)
        attention_weights: 注意力权重，形状为 (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # 计算注意力得分: Q @ K^T / sqrt(d_k)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # 应用mask
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax归一化
    attention_weights = softmax(scores, axis=-1)

    # 加权求和
    output = np.dot(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention:
    """
    多头注意力机制实现

    特点：
    - 使用多个注意力头并行计算
    - 每个头关注不同的表示子空间
    - 提升模型的表达能力和捕获多样化模式的能力
    - 参数量与单头注意力相同（通过减小每个头的维度）
    """

    def __init__(self, embed_dim, num_heads, use_bias=True):
        """
        初始化多头注意力层

        Args:
            embed_dim: 嵌入维度（必须能被num_heads整除）
            num_heads: 注意力头的数量
            use_bias: 是否使用偏置
        """
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.use_bias = use_bias

        # 初始化所有头的Q、K、V投影矩阵（合并成一个大矩阵以提高效率）
        # 也可以为每个头单独创建矩阵，但合并更高效
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # 输出投影矩阵W^O
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        if use_bias:
            self.b_q = np.zeros(embed_dim)
            self.b_k = np.zeros(embed_dim)
            self.b_v = np.zeros(embed_dim)
            self.b_o = np.zeros(embed_dim)

    def split_heads(self, x):
        """
        将输入分割成多个头

        Args:
            x: 输入，形状为 (seq_len, embed_dim)

        Returns:
            分割后的张量，形状为 (num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[0]
        # 重塑: (seq_len, embed_dim) -> (seq_len, num_heads, head_dim)
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        # 转置: (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim)
        x = x.transpose(1, 0, 2)
        return x

    def combine_heads(self, x):
        """
        将多个头的输出合并

        Args:
            x: 输入，形状为 (num_heads, seq_len, head_dim)

        Returns:
            合并后的张量，形状为 (seq_len, embed_dim)
        """
        # 转置: (num_heads, seq_len, head_dim) -> (seq_len, num_heads, head_dim)
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        # 重塑: (seq_len, num_heads, head_dim) -> (seq_len, embed_dim)
        x = x.reshape(seq_len, self.embed_dim)
        return x

    def forward(self, query, key=None, value=None, mask=None, return_attention=False):
        """
        前向传播

        Args:
            query: Query输入，形状为 (seq_len_q, embed_dim)
            key: Key输入，形状为 (seq_len_k, embed_dim)。如果为None，则使用query（自注意力）
            value: Value输入，形状为 (seq_len_v, embed_dim)。如果为None，则使用key
            mask: 注意力掩码，形状为 (seq_len_q, seq_len_k)
            return_attention: 是否返回注意力权重

        Returns:
            output: 输出，形状为 (seq_len_q, embed_dim)
            attention_weights: (可选) 所有头的注意力权重，形状为 (num_heads, seq_len_q, seq_len_k)
        """
        # 如果key和value为None，说明是自注意力
        if key is None:
            key = query
        if value is None:
            value = key

        seq_len_q = query.shape[0]
        seq_len_k = key.shape[0]

        # 步骤1: 线性投影得到Q、K、V
        Q = np.dot(query, self.W_q)  # (seq_len_q, embed_dim)
        K = np.dot(key, self.W_k)    # (seq_len_k, embed_dim)
        V = np.dot(value, self.W_v)  # (seq_len_v, embed_dim)

        if self.use_bias:
            Q += self.b_q
            K += self.b_k
            V += self.b_v

        # 步骤2: 分割成多个头
        Q = self.split_heads(Q)  # (num_heads, seq_len_q, head_dim)
        K = self.split_heads(K)  # (num_heads, seq_len_k, head_dim)
        V = self.split_heads(V)  # (num_heads, seq_len_v, head_dim)

        # 步骤3: 对每个头独立计算scaled dot-product attention
        head_outputs = []
        attention_weights_list = []

        for i in range(self.num_heads):
            # 取出第i个头的Q、K、V
            Q_i = Q[i]  # (seq_len_q, head_dim)
            K_i = K[i]  # (seq_len_k, head_dim)
            V_i = V[i]  # (seq_len_v, head_dim)

            # 计算attention
            head_output, attn_weights = scaled_dot_product_attention(
                Q_i, K_i, V_i, mask=mask
            )

            head_outputs.append(head_output)
            attention_weights_list.append(attn_weights)

        # 步骤4: 合并所有头的输出
        # 堆叠: list of (seq_len_q, head_dim) -> (num_heads, seq_len_q, head_dim)
        multi_head_output = np.stack(head_outputs, axis=0)

        # 合并头: (num_heads, seq_len_q, head_dim) -> (seq_len_q, embed_dim)
        concatenated = self.combine_heads(multi_head_output)

        # 步骤5: 通过输出投影W^O
        output = np.dot(concatenated, self.W_o)  # (seq_len_q, embed_dim)

        if self.use_bias:
            output += self.b_o

        if return_attention:
            # 返回所有头的注意力权重
            attention_weights = np.stack(attention_weights_list, axis=0)  # (num_heads, seq_len_q, seq_len_k)
            return output, attention_weights

        return output


class MultiHeadSelfAttention(MultiHeadAttention):
    """
    多头自注意力（Multi-Head Self-Attention）

    是MultiHeadAttention的特例，Q、K、V都来自同一个输入。
    主要用于Transformer的Encoder和Decoder中。
    """

    def forward(self, x, mask=None, return_attention=False):
        """
        前向传播（自注意力版本）

        Args:
            x: 输入序列，形状为 (seq_len, embed_dim)
            mask: 注意力掩码，形状为 (seq_len, seq_len)
            return_attention: 是否返回注意力权重

        Returns:
            output: 输出，形状为 (seq_len, embed_dim)
            attention_weights: (可选) 注意力权重
        """
        return super().forward(x, x, x, mask=mask, return_attention=return_attention)


def create_causal_mask(seq_len):
    """
    创建因果mask（用于GPT等自回归模型）

    Args:
        seq_len: 序列长度

    Returns:
        mask: 下三角矩阵，形状为 (seq_len, seq_len)
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def create_padding_mask(seq_lengths, max_len):
    """
    创建padding mask

    Args:
        seq_lengths: 每个序列的实际长度，形状为 (batch_size,)
        max_len: 最大序列长度

    Returns:
        mask: padding mask，形状为 (batch_size, max_len, max_len)
    """
    batch_size = len(seq_lengths)
    mask = np.zeros((batch_size, max_len, max_len))

    for i, length in enumerate(seq_lengths):
        mask[i, :length, :length] = 1

    return mask


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("多头注意力机制演示")
    print("=" * 70)

    # 参数设置
    seq_len = 10
    embed_dim = 512
    num_heads = 8

    print(f"\n配置:")
    print(f"  序列长度: {seq_len}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  每个头的维度: {embed_dim // num_heads}")

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 创建多头自注意力层
    mh_self_attn = MultiHeadSelfAttention(embed_dim, num_heads)

    # 前向传播（无mask）
    print("\n" + "=" * 70)
    print("1. 标准多头自注意力（无mask）")
    print("=" * 70)
    output, attn_weights = mh_self_attn.forward(x, return_attention=True)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"  - {num_heads}个头")
    print(f"  - 每个头的注意力矩阵: ({seq_len}, {seq_len})")

    # 验证每个头的注意力权重和为1
    print(f"\n各头的注意力权重和（应该都接近1.0）:")
    for i in range(num_heads):
        row_sums = np.sum(attn_weights[i], axis=1)
        print(f"  头{i}: {row_sums[0]:.6f}")

    # 使用因果mask（GPT风格）
    print("\n" + "=" * 70)
    print("2. 带因果mask的多头自注意力（GPT风格）")
    print("=" * 70)
    causal_mask = create_causal_mask(seq_len)
    print(f"因果mask形状: {causal_mask.shape}")
    print(f"因果mask（前5行）:\n{causal_mask[:5, :5].astype(int)}")

    output_masked, attn_weights_masked = mh_self_attn.forward(
        x, mask=causal_mask, return_attention=True
    )

    print(f"\n带mask的输出形状: {output_masked.shape}")
    print(f"带mask的注意力权重形状: {attn_weights_masked.shape}")

    # 显示第一个头的mask效果
    print(f"\n第1个头的注意力权重（前5行，带因果mask）:")
    print(attn_weights_masked[0, :5, :5])
    print("注意：每个位置只能看到自己和之前的位置")

    # 跨注意力示例（Cross-Attention）
    print("\n" + "=" * 70)
    print("3. 多头跨注意力（Cross-Attention）")
    print("=" * 70)

    # 模拟encoder-decoder场景
    encoder_seq_len = 12
    decoder_seq_len = 8
    encoder_output = np.random.randn(encoder_seq_len, embed_dim)
    decoder_input = np.random.randn(decoder_seq_len, embed_dim)

    mh_cross_attn = MultiHeadAttention(embed_dim, num_heads)

    # Query来自decoder，Key和Value来自encoder
    cross_output, cross_attn_weights = mh_cross_attn.forward(
        query=decoder_input,
        key=encoder_output,
        value=encoder_output,
        return_attention=True
    )

    print(f"Decoder输入（Query）: {decoder_input.shape}")
    print(f"Encoder输出（Key/Value）: {encoder_output.shape}")
    print(f"跨注意力输出: {cross_output.shape}")
    print(f"跨注意力权重: {cross_attn_weights.shape}")
    print(f"  - Decoder的{decoder_seq_len}个位置关注Encoder的{encoder_seq_len}个位置")

    # 分析不同头的注意力模式
    print("\n" + "=" * 70)
    print("4. 多头注意力的优势分析")
    print("=" * 70)

    print("\n为什么要用多头？")
    print("  ✓ 不同的头可以关注不同的模式（句法、语义等）")
    print("  ✓ 增强模型的表达能力")
    print("  ✓ 类似于CNN中的多个卷积核")
    print("  ✓ 每个头的维度更小，计算效率高")

    # 计算参数量
    total_params = (
        embed_dim * embed_dim * 3 +  # W_q, W_k, W_v
        embed_dim * embed_dim        # W_o
    )
    if mh_self_attn.use_bias:
        total_params += embed_dim * 4  # b_q, b_k, b_v, b_o

    print(f"\n参数量分析:")
    print(f"  W_q, W_k, W_v: {embed_dim}×{embed_dim} × 3 = {embed_dim * embed_dim * 3:,}")
    print(f"  W_o: {embed_dim}×{embed_dim} = {embed_dim * embed_dim:,}")
    print(f"  总参数量: {total_params:,}")

    # 复杂度分析
    print(f"\n时间复杂度:")
    print(f"  投影Q、K、V: O(n·d²) = O({seq_len}·{embed_dim}²)")
    print(f"  多头注意力: O(h·n²·d/h) = O(n²·d) = O({seq_len}²·{embed_dim})")
    print(f"  输出投影: O(n·d²) = O({seq_len}·{embed_dim}²)")
    print(f"  总计: O(n²·d + n·d²)")

    # 与单头注意力对比
    print("\n" + "=" * 70)
    print("5. 多头 vs 单头注意力")
    print("=" * 70)

    # 创建单头注意力（num_heads=1）
    single_head_attn = MultiHeadSelfAttention(embed_dim, num_heads=1)
    output_single, attn_weights_single = single_head_attn.forward(x, return_attention=True)

    print(f"\n单头注意力:")
    print(f"  头数: 1")
    print(f"  每个头维度: {embed_dim}")
    print(f"  注意力权重形状: {attn_weights_single.shape}")

    print(f"\n多头注意力:")
    print(f"  头数: {num_heads}")
    print(f"  每个头维度: {embed_dim // num_heads}")
    print(f"  注意力权重形状: {attn_weights.shape}")

    print("\n对比:")
    print(f"  ✓ 两者参数量相同: {total_params:,}")
    print(f"  ✓ 多头可以捕获{num_heads}种不同的注意力模式")
    print(f"  ✓ 多头的表达能力更强")

    print("\n" + "=" * 70)
    print("多头注意力的关键特性:")
    print("=" * 70)
    print("✓ 并行计算多个注意力头，捕获不同子空间的信息")
    print("✓ 每个头关注不同的表示模式（位置、语义、句法等）")
    print("✓ 参数量与单头相同，但表达能力更强")
    print("✓ 是Transformer架构的核心组件")
    print("✓ 广泛应用于BERT、GPT、T5等所有现代LLM")
