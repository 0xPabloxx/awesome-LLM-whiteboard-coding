"""
Self-Attention (自注意力) 实现

自注意力是Transformer的核心机制，序列内部元素相互计算注意力。
每个位置都会关注序列中的所有位置（包括自己），捕获序列内部的依赖关系。

核心思想：
1. 从输入生成Query、Key、Value三个矩阵
2. 计算Query和Key的点积得到注意力得分
3. 缩放并Softmax归一化
4. 用注意力权重对Value加权求和

应用：Transformer、BERT、GPT等所有现代LLM的基础
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SelfAttention:
    """
    自注意力机制实现

    特点：
    - Query、Key、Value都来自同一个输入序列
    - 每个位置可以关注所有位置（包括自己）
    - 捕获序列内部的长距离依赖
    - 计算复杂度为O(n²)，n为序列长度
    """

    def __init__(self, embed_dim, use_bias=True):
        """
        初始化自注意力层

        Args:
            embed_dim: 嵌入维度
            use_bias: 是否使用偏置
        """
        self.embed_dim = embed_dim
        self.use_bias = use_bias

        # 初始化Q、K、V的投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        if use_bias:
            self.b_q = np.zeros(embed_dim)
            self.b_k = np.zeros(embed_dim)
            self.b_v = np.zeros(embed_dim)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入序列，形状为 (seq_len, embed_dim)
            mask: 注意力掩码，形状为 (seq_len, seq_len)，
                  用于屏蔽某些位置（如padding或未来信息）

        Returns:
            output: 输出序列，形状为 (seq_len, embed_dim)
            attention_weights: 注意力权重，形状为 (seq_len, seq_len)
        """
        seq_len, embed_dim = x.shape

        # 步骤1: 线性投影得到Q、K、V
        Q = np.dot(x, self.W_q)  # (seq_len, embed_dim)
        K = np.dot(x, self.W_k)  # (seq_len, embed_dim)
        V = np.dot(x, self.W_v)  # (seq_len, embed_dim)

        if self.use_bias:
            Q += self.b_q
            K += self.b_k
            V += self.b_v

        # 步骤2: 计算注意力得分（Scaled Dot-Product）
        # Q @ K^T / sqrt(d_k)
        scores = np.dot(Q, K.T) / np.sqrt(embed_dim)  # (seq_len, seq_len)

        # 步骤3: 应用mask（如果提供）
        if mask is not None:
            # mask中为0的位置设置为负无穷，softmax后会变成0
            scores = np.where(mask == 0, -1e9, scores)

        # 步骤4: Softmax归一化得到注意力权重
        attention_weights = softmax(scores, axis=-1)  # (seq_len, seq_len)

        # 步骤5: 加权求和Value
        output = np.dot(attention_weights, V)  # (seq_len, embed_dim)

        return output, attention_weights


def self_attention_simple(x):
    """
    简化版自注意力（不使用可学习参数）

    Args:
        x: 输入序列，形状为 (seq_len, embed_dim)

    Returns:
        output: 输出序列，形状为 (seq_len, embed_dim)
        weights: 注意力权重，形状为 (seq_len, seq_len)
    """
    seq_len, embed_dim = x.shape

    # 直接使用输入作为Q、K、V
    Q = K = V = x

    # 计算注意力得分
    scores = np.dot(Q, K.T) / np.sqrt(embed_dim)

    # Softmax
    weights = softmax(scores, axis=-1)

    # 加权求和
    output = np.dot(weights, V)

    return output, weights


def create_causal_mask(seq_len):
    """
    创建因果mask（用于GPT等自回归模型）

    Args:
        seq_len: 序列长度

    Returns:
        mask: 下三角矩阵，形状为 (seq_len, seq_len)
    """
    # 下三角矩阵：当前位置只能看到之前的位置
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

    print("=" * 60)
    print("自注意力机制演示")
    print("=" * 60)

    # 参数设置
    seq_len = 8
    embed_dim = 64

    # 生成示例输入序列
    x = np.random.randn(seq_len, embed_dim)

    # 创建自注意力层
    self_attn = SelfAttention(embed_dim)

    # 前向传播（无mask）
    print("\n1. 标准自注意力（无mask）")
    output, attn_weights = self_attn.forward(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"\n注意力权重矩阵（每行是一个位置关注所有位置的权重）:")
    print(attn_weights[:3, :])  # 打印前3行
    print(f"\n每行权重和: {np.sum(attn_weights, axis=1)[:3]}")  # 应该都是1.0

    # 使用因果mask（GPT风格）
    print("\n" + "=" * 60)
    print("2. 带因果mask的自注意力（GPT风格）")
    causal_mask = create_causal_mask(seq_len)
    print(f"因果mask:\n{causal_mask.astype(int)}")

    output_masked, attn_weights_masked = self_attn.forward(x, mask=causal_mask)
    print(f"\n带mask的注意力权重（前3行）:")
    print(attn_weights_masked[:3, :])
    print("注意：每个位置只能看到自己和之前的位置")

    # 简化版自注意力
    print("\n" + "=" * 60)
    print("3. 简化版自注意力（无参数）")
    output_simple, weights_simple = self_attention_simple(x)
    print(f"输出形状: {output_simple.shape}")
    print(f"注意力权重形状: {weights_simple.shape}")

    # 可视化一个位置的注意力
    print("\n" + "=" * 60)
    print("4. 注意力权重可视化")
    position = 3
    print(f"\n位置 {position} 对所有位置的注意力权重:")
    print(f"权重: {attn_weights[position]}")
    print(f"最关注的位置: {np.argmax(attn_weights[position])}")
    print(f"对自己的注意力: {attn_weights[position, position]:.4f}")

    print("\n" + "=" * 60)
    print("自注意力的关键特性:")
    print("=" * 60)
    print("✓ 序列内部的每个元素都关注所有元素（包括自己）")
    print("✓ 可以捕获长距离依赖关系")
    print("✓ 并行计算，不依赖递归")
    print("✓ 位置之间的交互是对称的（可以看作全连接图）")
    print("✗ 计算复杂度为O(n²)，序列长时开销大")
