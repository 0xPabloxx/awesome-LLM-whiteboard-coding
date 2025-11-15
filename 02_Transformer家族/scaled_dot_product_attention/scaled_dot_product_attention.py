"""
Scaled Dot-Product Attention (缩放点积注意力) 实现

缩放点积注意力是Transformer模型的核心组件，由Vaswani等人在2017年提出。
它是Multi-Head Attention的基础构建块，通过缩放因子解决了点积值过大的问题。

核心思想：
1. 计算查询(Q)和键(K)的点积得到注意力分数
2. 用sqrt(d_k)进行缩放，防止点积值过大导致softmax梯度消失
3. 应用softmax得到注意力权重
4. 用权重对值(V)进行加权求和

公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
"""

import numpy as np


def softmax(x, axis=-1):
    """
    Softmax函数实现

    Args:
        x: 输入数组
        axis: 进行softmax的轴
    Returns:
        softmax归一化后的数组
    """
    # 减去最大值提高数值稳定性，避免指数溢出
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    """
    缩放点积注意力机制实现类

    特点：
    - 使用点积计算相似度，计算效率高
    - 通过sqrt(d_k)缩放，避免点积值过大
    - 支持mask机制，用于padding和因果注意力
    - 可批量处理，支持矩阵运算
    """

    def __init__(self, temperature=None):
        """
        初始化缩放点积注意力层

        Args:
            temperature: 温度参数，默认为None时使用sqrt(d_k)
                        也可以手动设置其他缩放因子
        """
        self.temperature = temperature

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        Args:
            Q: 查询矩阵，形状为 (batch_size, seq_len_q, d_k) 或 (seq_len_q, d_k)
            K: 键矩阵，形状为 (batch_size, seq_len_k, d_k) 或 (seq_len_k, d_k)
            V: 值矩阵，形状为 (batch_size, seq_len_v, d_v) 或 (seq_len_v, d_v)
               注意：seq_len_k 必须等于 seq_len_v
            mask: 掩码矩阵，形状为 (seq_len_q, seq_len_k) 或 (batch_size, seq_len_q, seq_len_k)
                 值为True的位置会被mask掉（设为-inf）

        Returns:
            output: 注意力输出，形状与V相同
            attention_weights: 注意力权重，形状为 (batch_size, seq_len_q, seq_len_k) 或 (seq_len_q, seq_len_k)
        """
        # 获取d_k维度（键的维度）
        d_k = K.shape[-1]

        # 步骤1: 计算注意力分数 QK^T
        # Q: (..., seq_len_q, d_k), K: (..., seq_len_k, d_k)
        # scores: (..., seq_len_q, seq_len_k)
        scores = np.matmul(Q, np.swapaxes(K, -2, -1))

        # 步骤2: 缩放，除以sqrt(d_k)
        # 为什么要缩放？
        # - 当d_k较大时，点积的值会很大
        # - 导致softmax函数进入饱和区，梯度接近0
        # - 缩放可以保持方差稳定，避免梯度消失
        if self.temperature is None:
            scores = scores / np.sqrt(d_k)
        else:
            scores = scores / self.temperature

        # 步骤3: 应用mask（可选）
        if mask is not None:
            # 将mask为True的位置设为一个很小的负数
            # 这样softmax后这些位置的权重接近0
            scores = np.where(mask, -1e9, scores)

        # 步骤4: Softmax归一化得到注意力权重
        # 对最后一维（seq_len_k）进行softmax
        attention_weights = softmax(scores, axis=-1)

        # 步骤5: 用注意力权重对V进行加权求和
        # attention_weights: (..., seq_len_q, seq_len_k)
        # V: (..., seq_len_k, d_v)
        # output: (..., seq_len_q, d_v)
        output = np.matmul(attention_weights, V)

        return output, attention_weights


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力的简化函数实现（不使用类）

    Args:
        Q: 查询矩阵，形状为 (..., seq_len_q, d_k)
        K: 键矩阵，形状为 (..., seq_len_k, d_k)
        V: 值矩阵，形状为 (..., seq_len_k, d_v)
        mask: 掩码矩阵（可选）

    Returns:
        output: 注意力输出
        attention_weights: 注意力权重
    """
    d_k = K.shape[-1]

    # QK^T / sqrt(d_k)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)

    # 应用mask
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Softmax
    attention_weights = softmax(scores, axis=-1)

    # 加权求和
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def create_padding_mask(seq_len, padding_positions):
    """
    创建padding mask

    Args:
        seq_len: 序列长度
        padding_positions: padding位置的列表

    Returns:
        mask: padding mask矩阵
    """
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for pos in padding_positions:
        mask[:, pos] = True  # 对所有query位置，mask掉padding的key
    return mask


def create_causal_mask(seq_len):
    """
    创建因果mask（用于decoder自注意力）

    防止当前位置attend到未来位置的信息

    Args:
        seq_len: 序列长度

    Returns:
        mask: 因果mask矩阵，上三角部分为True
    """
    # 创建上三角矩阵（不包括对角线）
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return mask


# 示例使用
if __name__ == "__main__":
    # 设置随机种子以便复现
    np.random.seed(42)

    print("=" * 60)
    print("缩放点积注意力机制演示")
    print("=" * 60)

    # 示例1: 基础使用
    print("\n【示例1】基础使用 - 单个序列")
    print("-" * 60)

    seq_len = 8
    d_k = 64  # 键/查询的维度
    d_v = 64  # 值的维度

    # 生成随机的Q, K, V矩阵
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    # 创建注意力层
    attention = ScaledDotProductAttention()

    # 前向传播
    output, weights = attention.forward(Q, K, V)

    print(f"查询矩阵Q形状: {Q.shape}")
    print(f"键矩阵K形状: {K.shape}")
    print(f"值矩阵V形状: {V.shape}")
    print(f"\n输出矩阵形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    print(f"\n注意力权重矩阵（前3行前5列）:")
    print(weights[:3, :5])
    print(f"\n每行权重之和（应该都接近1.0）: {weights.sum(axis=-1)[:5]}")

    # 示例2: 批量处理
    print("\n" + "=" * 60)
    print("【示例2】批量处理 - 多个序列")
    print("-" * 60)

    batch_size = 4
    seq_len = 10
    d_k = 64
    d_v = 64

    # 生成批量数据
    Q_batch = np.random.randn(batch_size, seq_len, d_k)
    K_batch = np.random.randn(batch_size, seq_len, d_k)
    V_batch = np.random.randn(batch_size, seq_len, d_v)

    # 使用简化函数
    output_batch, weights_batch = scaled_dot_product_attention(Q_batch, K_batch, V_batch)

    print(f"批量查询矩阵Q形状: {Q_batch.shape}")
    print(f"批量输出矩阵形状: {output_batch.shape}")
    print(f"批量注意力权重形状: {weights_batch.shape}")

    # 示例3: 使用Padding Mask
    print("\n" + "=" * 60)
    print("【示例3】Padding Mask - 处理变长序列")
    print("-" * 60)

    seq_len = 6
    # 假设最后2个位置是padding
    padding_positions = [4, 5]

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    # 创建padding mask
    padding_mask = create_padding_mask(seq_len, padding_positions)
    print(f"Padding mask矩阵（位置{padding_positions}被mask）:")
    print(padding_mask.astype(int))

    # 应用mask
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=padding_mask)

    print(f"\n应用mask后的注意力权重（前3行）:")
    print(weights_masked[:3])
    print(f"\n注意：位置{padding_positions}的权重应该接近0")

    # 示例4: 使用Causal Mask（因果mask）
    print("\n" + "=" * 60)
    print("【示例4】Causal Mask - 用于Decoder自注意力")
    print("-" * 60)

    seq_len = 5
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    # 创建因果mask
    causal_mask = create_causal_mask(seq_len)
    print(f"因果mask矩阵（上三角为True，防止看到未来信息）:")
    print(causal_mask.astype(int))

    # 应用因果mask
    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    print(f"\n应用因果mask后的注意力权重:")
    print(weights_causal)
    print(f"\n注意：每行只能attend到当前及之前的位置")

    # 示例5: 对比缩放的影响
    print("\n" + "=" * 60)
    print("【示例5】缩放的重要性")
    print("-" * 60)

    d_k = 512  # 使用较大的维度来展示缩放的效果
    Q = np.random.randn(5, d_k)
    K = np.random.randn(5, d_k)
    V = np.random.randn(5, d_k)

    # 不缩放的注意力分数
    scores_no_scale = np.matmul(Q, np.swapaxes(K, -2, -1))
    print(f"维度d_k = {d_k}")
    print(f"不缩放的分数范围: [{scores_no_scale.min():.2f}, {scores_no_scale.max():.2f}]")
    print(f"不缩放的分数方差: {scores_no_scale.var():.2f}")

    # 缩放后的注意力分数
    scores_scaled = scores_no_scale / np.sqrt(d_k)
    print(f"\n缩放后的分数范围: [{scores_scaled.min():.2f}, {scores_scaled.max():.2f}]")
    print(f"缩放后的分数方差: {scores_scaled.var():.2f}")

    # 比较softmax后的分布
    weights_no_scale = softmax(scores_no_scale, axis=-1)
    weights_scaled = softmax(scores_scaled, axis=-1)

    print(f"\n不缩放的权重最大值: {weights_no_scale.max():.4f}")
    print(f"缩放后的权重最大值: {weights_scaled.max():.4f}")
    print(f"\n结论：缩放使得注意力分布更加平滑，避免过度集中")

    print("\n" + "=" * 60)
