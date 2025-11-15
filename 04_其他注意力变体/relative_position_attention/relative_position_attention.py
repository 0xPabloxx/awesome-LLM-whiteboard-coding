"""
Relative Position Attention (相对位置注意力) 实现

相对位置注意力是一种改进的注意力机制，它编码token之间的相对位置关系，而非绝对位置。
由Shaw等人在2018年的论文"Self-Attention with Relative Position Representations"中提出，
并在T5、Transformer-XL等模型中被广泛采用。

核心思想：
1. 使用相对位置而非绝对位置编码
2. 捕获token之间的相对距离关系
3. 更好的长度泛化能力
4. 支持任意长度的序列

公式：
对于位置i和j，注意力计算为：
  score(i,j) = (Q_i · K_j + Q_i · R_{i-j}) / sqrt(d_k)

其中R_{i-j}是相对位置编码，只依赖于相对距离i-j

优势：
- 位置不变性：只关注相对距离
- 长度泛化：训练时的长度可以不同于推理时
- 更自然：语言理解更依赖相对位置而非绝对位置

应用：T5、Transformer-XL、DeBERTa等模型
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def get_relative_positions(seq_len, max_relative_position=None):
    """
    生成相对位置矩阵

    Args:
        seq_len: 序列长度
        max_relative_position: 最大相对位置距离。如果为None，则为seq_len-1

    Returns:
        relative_positions: 相对位置矩阵，形状为 (seq_len, seq_len)

    示例：
        seq_len=4时，relative_positions为：
        [[ 0, -1, -2, -3],
         [ 1,  0, -1, -2],
         [ 2,  1,  0, -1],
         [ 3,  2,  1,  0]]

        如果max_relative_position=2，则裁剪为：
        [[ 0, -1, -2, -2],
         [ 1,  0, -1, -2],
         [ 2,  1,  0, -1],
         [ 2,  2,  1,  0]]
    """
    if max_relative_position is None:
        max_relative_position = seq_len - 1

    # 创建位置索引矩阵
    range_vec = np.arange(seq_len)
    # 计算相对位置：range_vec[i] - range_vec[j]
    relative_positions = range_vec[None, :] - range_vec[:, None]

    # 裁剪到[-max_relative_position, max_relative_position]
    relative_positions = np.clip(
        relative_positions,
        -max_relative_position,
        max_relative_position
    )

    return relative_positions


def relative_position_to_index(relative_positions, max_relative_position):
    """
    将相对位置转换为嵌入索引

    Args:
        relative_positions: 相对位置矩阵，值范围[-max_relative_position, max_relative_position]
        max_relative_position: 最大相对位置

    Returns:
        indices: 嵌入索引，值范围[0, 2*max_relative_position]

    说明：
        相对位置-k映射到索引max_relative_position-k
        相对位置0映射到索引max_relative_position
        相对位置+k映射到索引max_relative_position+k
    """
    # 将[-max_relative_position, max_relative_position]映射到[0, 2*max_relative_position]
    indices = relative_positions + max_relative_position
    return indices


class RelativePositionAttention:
    """
    Relative Position Attention (相对位置注意力)

    特点：
    - 使用相对位置编码而非绝对位置编码
    - 注意力计算同时考虑内容和相对位置
    - 更好的长度泛化能力
    - 参数量与序列长度无关（使用max_relative_position）
    """

    def __init__(self, embed_dim, num_heads, max_relative_position=None, use_bias=True):
        """
        初始化相对位置注意力层

        Args:
            embed_dim: 嵌入维度（必须能被num_heads整除）
            num_heads: 注意力头的数量
            max_relative_position: 最大相对位置距离。如果为None，则不限制
            use_bias: 是否使用偏置
        """
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_position = max_relative_position
        self.use_bias = use_bias

        # Q, K, V投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # 输出投影矩阵
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        if use_bias:
            self.b_q = np.zeros(embed_dim)
            self.b_k = np.zeros(embed_dim)
            self.b_v = np.zeros(embed_dim)
            self.b_o = np.zeros(embed_dim)

        # 相对位置嵌入
        # 如果max_relative_position未指定，我们会在forward中动态创建
        if max_relative_position is not None:
            # 需要2*max_relative_position+1个不同的相对位置嵌入
            # （-max_relative_position到+max_relative_position）
            num_relative_positions = 2 * max_relative_position + 1
            # 每个头都有自己的相对位置嵌入
            self.relative_position_embeddings = np.random.randn(
                num_heads, num_relative_positions, self.head_dim
            ) / np.sqrt(self.head_dim)
        else:
            self.relative_position_embeddings = None

    def _get_relative_embeddings(self, seq_len):
        """
        获取或创建相对位置嵌入

        Args:
            seq_len: 序列长度

        Returns:
            relative_embeddings: 相对位置嵌入
        """
        max_relative_position = self.max_relative_position
        if max_relative_position is None:
            max_relative_position = seq_len - 1

        if self.relative_position_embeddings is None:
            # 动态创建
            num_relative_positions = 2 * max_relative_position + 1
            relative_embeddings = np.random.randn(
                self.num_heads, num_relative_positions, self.head_dim
            ) / np.sqrt(self.head_dim)
        else:
            relative_embeddings = self.relative_position_embeddings

        return relative_embeddings, max_relative_position

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

    def compute_attention_with_relative_positions(self, Q, K, V, relative_embeddings,
                                                   relative_positions, mask=None):
        """
        计算带相对位置编码的注意力

        Args:
            Q: Query，形状为 (num_heads, seq_len_q, head_dim)
            K: Key，形状为 (num_heads, seq_len_k, head_dim)
            V: Value，形状为 (num_heads, seq_len_v, head_dim)
            relative_embeddings: 相对位置嵌入，形状为 (num_heads, num_relative_positions, head_dim)
            relative_positions: 相对位置矩阵，形状为 (seq_len_q, seq_len_k)
            mask: 注意力掩码

        Returns:
            output: 输出，形状为 (num_heads, seq_len_q, head_dim)
            attention_weights: 注意力权重，形状为 (num_heads, seq_len_q, seq_len_k)
        """
        num_heads, seq_len_q, head_dim = Q.shape
        seq_len_k = K.shape[1]

        # 1. 计算内容得分: Q @ K^T
        # (num_heads, seq_len_q, head_dim) @ (num_heads, head_dim, seq_len_k)
        # -> (num_heads, seq_len_q, seq_len_k)
        content_scores = np.matmul(Q, K.transpose(0, 2, 1))

        # 2. 计算相对位置得分
        # 获取相对位置嵌入的索引
        max_relative_position = (relative_embeddings.shape[1] - 1) // 2
        relative_indices = relative_position_to_index(relative_positions, max_relative_position)

        # 对每个头计算相对位置得分
        position_scores = np.zeros((num_heads, seq_len_q, seq_len_k))
        for h in range(num_heads):
            # 获取该头的相对位置嵌入
            # relative_embeddings[h]: (num_relative_positions, head_dim)
            # relative_indices: (seq_len_q, seq_len_k)

            # 对于每个query位置i和key位置j，获取相对位置嵌入R_{i-j}
            for i in range(seq_len_q):
                for j in range(seq_len_k):
                    rel_idx = relative_indices[i, j]
                    # Q[h, i]: (head_dim,)
                    # relative_embeddings[h, rel_idx]: (head_dim,)
                    # 点积得到标量
                    position_scores[h, i, j] = np.dot(Q[h, i], relative_embeddings[h, rel_idx])

        # 3. 合并内容得分和位置得分
        scores = (content_scores + position_scores) / np.sqrt(head_dim)

        # 4. 应用mask
        if mask is not None:
            # mask的形状可能是(seq_len_q, seq_len_k)，需要扩展到(num_heads, seq_len_q, seq_len_k)
            if mask.ndim == 2:
                mask = mask[None, :, :]  # 广播到所有头
            scores = np.where(mask == 0, -1e9, scores)

        # 5. Softmax归一化
        attention_weights = softmax(scores, axis=-1)

        # 6. 加权求和V
        # (num_heads, seq_len_q, seq_len_k) @ (num_heads, seq_len_k, head_dim)
        # -> (num_heads, seq_len_q, head_dim)
        output = np.matmul(attention_weights, V)

        return output, attention_weights

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
            attention_weights: (可选) 注意力权重
        """
        # 如果key和value为None，说明是自注意力
        if key is None:
            key = query
        if value is None:
            value = key

        seq_len_q = query.shape[0]
        seq_len_k = key.shape[0]

        # 1. 线性投影得到Q、K、V
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)

        if self.use_bias:
            Q += self.b_q
            K += self.b_k
            V += self.b_v

        # 2. 分割成多个头
        Q = self.split_heads(Q)  # (num_heads, seq_len_q, head_dim)
        K = self.split_heads(K)  # (num_heads, seq_len_k, head_dim)
        V = self.split_heads(V)  # (num_heads, seq_len_v, head_dim)

        # 3. 获取相对位置信息
        relative_embeddings, max_relative_position = self._get_relative_embeddings(
            max(seq_len_q, seq_len_k)
        )
        relative_positions = get_relative_positions(seq_len_q, max_relative_position)
        if seq_len_q != seq_len_k:
            # 处理交叉注意力的情况
            relative_positions = relative_positions[:, :seq_len_k]

        # 4. 计算带相对位置的注意力
        multi_head_output, attention_weights = self.compute_attention_with_relative_positions(
            Q, K, V, relative_embeddings, relative_positions, mask=mask
        )

        # 5. 合并头
        concatenated = self.combine_heads(multi_head_output)

        # 6. 输出投影
        output = np.dot(concatenated, self.W_o)
        if self.use_bias:
            output += self.b_o

        if return_attention:
            return output, attention_weights

        return output


class RelativePositionSelfAttention(RelativePositionAttention):
    """
    Relative Position Self-Attention (相对位置自注意力)

    是RelativePositionAttention的特例，Q、K、V都来自同一个输入。
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


def visualize_relative_positions(seq_len, max_relative_position=None):
    """
    可视化相对位置矩阵

    Args:
        seq_len: 序列长度
        max_relative_position: 最大相对位置
    """
    relative_positions = get_relative_positions(seq_len, max_relative_position)

    print(f"相对位置矩阵 (seq_len={seq_len}, max_relative_position={max_relative_position}):")
    print(relative_positions)
    print("\n说明:")
    print("  - 矩阵[i,j]表示从位置i到位置j的相对距离")
    print("  - 正值表示j在i之后，负值表示j在i之前")
    print("  - 对角线为0（自身到自身）")


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("Relative Position Attention (相对位置注意力) 演示")
    print("=" * 80)

    # 参数设置
    seq_len = 8
    embed_dim = 64
    num_heads = 4
    max_relative_position = 4

    print(f"\n配置:")
    print(f"  序列长度: {seq_len}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  最大相对位置: {max_relative_position}")
    print(f"  每个头的维度: {embed_dim // num_heads}")

    # 1. 可视化相对位置
    print("\n" + "=" * 80)
    print("1. 相对位置矩阵")
    print("=" * 80)
    visualize_relative_positions(seq_len, max_relative_position)

    # 2. 创建相对位置注意力层
    print("\n" + "=" * 80)
    print("2. 相对位置注意力计算")
    print("=" * 80)

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 创建相对位置注意力层
    rpa = RelativePositionSelfAttention(embed_dim, num_heads, max_relative_position)

    # 前向传播
    output, attn_weights = rpa.forward(x, return_attention=True)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"  - {num_heads}个注意力头")
    print(f"  - 每个头的注意力矩阵: ({seq_len}, {seq_len})")

    # 3. 显示注意力模式
    print("\n" + "=" * 80)
    print("3. 注意力权重示例（第1个头）")
    print("=" * 80)
    print("\n注意力权重矩阵（行=query位置，列=key位置）:")
    print(attn_weights[0])
    print("\n说明:")
    print("  - 每行和为1.0（softmax归一化）")
    print("  - 相对位置编码影响注意力分布")

    # 4. 因果mask示例
    print("\n" + "=" * 80)
    print("4. 带因果mask的相对位置注意力（GPT风格）")
    print("=" * 80)

    causal_mask = create_causal_mask(seq_len)
    output_masked, attn_weights_masked = rpa.forward(x, mask=causal_mask, return_attention=True)

    print(f"\n因果mask形状: {causal_mask.shape}")
    print("因果mask（只能看到当前及之前的位置）:")
    print(causal_mask.astype(int))

    print("\n带因果mask的注意力权重（第1个头）:")
    print(attn_weights_masked[0])
    print("\n观察:")
    print("  - 每个位置只关注自己和之前的位置")
    print("  - 未来位置的注意力权重为0")

    # 5. 对比不同序列长度的泛化能力
    print("\n" + "=" * 80)
    print("5. 长度泛化能力测试")
    print("=" * 80)

    # 测试更长的序列
    longer_seq_len = 12
    x_longer = np.random.randn(longer_seq_len, embed_dim)

    print(f"\n训练长度: {seq_len}")
    print(f"测试长度: {longer_seq_len}")

    # 使用相同的相对位置注意力层
    output_longer, attn_weights_longer = rpa.forward(x_longer, return_attention=True)

    print(f"\n输入形状: {x_longer.shape}")
    print(f"输出形状: {output_longer.shape}")
    print(f"注意力权重形状: {attn_weights_longer.shape}")
    print("\n✓ 相对位置注意力可以处理不同长度的序列")
    print("✓ max_relative_position限制了需要学习的位置嵌入数量")

    # 6. 分析相对位置的影响
    print("\n" + "=" * 80)
    print("6. 相对位置对注意力的影响")
    print("=" * 80)

    # 选择一个query位置，查看它对不同相对距离的key的注意力
    query_pos = 4
    print(f"\nQuery位置{query_pos}对所有Key位置的注意力:")
    print(f"{'Key位置':>10} {'相对距离':>10} {'注意力权重':>12}")
    print("-" * 35)

    relative_positions = get_relative_positions(seq_len, max_relative_position)
    for key_pos in range(seq_len):
        rel_dist = relative_positions[query_pos, key_pos]
        attn_weight = attn_weights[0, query_pos, key_pos]
        print(f"{key_pos:>10} {rel_dist:>10} {attn_weight:>12.4f}")

    # 7. T5风格的相对位置注意力
    print("\n" + "=" * 80)
    print("7. T5风格配置")
    print("=" * 80)

    # T5使用较小的max_relative_position（通常为32或128）
    t5_max_relative_position = 32
    t5_rpa = RelativePositionSelfAttention(
        embed_dim=512,
        num_heads=8,
        max_relative_position=t5_max_relative_position
    )

    # T5的相对位置嵌入数量
    num_relative_embeddings = 2 * t5_max_relative_position + 1

    print(f"\nT5配置:")
    print(f"  嵌入维度: 512")
    print(f"  注意力头数: 8")
    print(f"  最大相对位置: {t5_max_relative_position}")
    print(f"  相对位置嵌入数量: {num_relative_embeddings}")
    print(f"  每个头的位置嵌入参数: {num_relative_embeddings} × {512//8} = {num_relative_embeddings * (512//8):,}")

    print("\nT5的设计选择:")
    print("  ✓ 使用有限的max_relative_position（通常32或128）")
    print("  ✓ 超过此距离的位置共享相同的嵌入")
    print("  ✓ 减少参数量，同时保持长距离建模能力")
    print("  ✓ 实验表明这足以捕获大多数有用的相对位置信息")

    # 8. 与绝对位置编码的对比
    print("\n" + "=" * 80)
    print("8. 相对位置 vs 绝对位置编码")
    print("=" * 80)

    print("\n相对位置注意力的优势:")
    print("  ✓ 位置不变性：模式只依赖相对距离")
    print("  ✓ 长度泛化：可以处理比训练时更长的序列")
    print("  ✓ 参数效率：使用max_relative_position限制参数量")
    print("  ✓ 自然建模：语言理解更依赖相对位置")

    print("\n绝对位置编码的问题:")
    print("  ✗ 序列长度受限于训练时的最大长度")
    print("  ✗ 需要为每个可能的位置学习嵌入")
    print("  ✗ 泛化能力较弱")

    print("\n实际应用:")
    print("  • T5: 使用相对位置注意力，max_relative_position=128")
    print("  • Transformer-XL: 使用相对位置编码处理长序列")
    print("  • DeBERTa: 使用解耦的内容和位置注意力")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("相对位置注意力通过编码相对距离而非绝对位置，")
    print("提供了更好的长度泛化能力和更自然的语言建模方式。")
    print("这是现代Transformer架构的重要改进之一。")
