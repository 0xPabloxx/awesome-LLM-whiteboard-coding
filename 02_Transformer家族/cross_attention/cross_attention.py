"""
Cross-Attention (交叉注意力) 实现

交叉注意力用于连接编码器和解码器，是Seq2Seq模型的关键组件。
与自注意力不同，Query来自解码器，而Key和Value来自编码器。

核心思想：
1. Query从解码器（目标序列）生成
2. Key和Value从编码器（源序列）生成
3. 解码器的每个位置可以关注编码器的所有位置
4. 实现源序列和目标序列之间的信息交互

应用：机器翻译、图像描述生成、语音识别等所有Encoder-Decoder架构
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class CrossAttention:
    """
    交叉注意力机制实现

    特点：
    - Query来自解码器（decoder，目标序列）
    - Key和Value来自编码器（encoder，源序列）
    - 解码器每个位置可以关注编码器的所有位置
    - 实现encoder-decoder之间的信息流动

    与Self-Attention的区别：
    - Self-Attention: Q、K、V都来自同一个序列
    - Cross-Attention: Q来自一个序列，K、V来自另一个序列
    """

    def __init__(self, embed_dim, use_bias=True):
        """
        初始化交叉注意力层

        Args:
            embed_dim: 嵌入维度（假设encoder和decoder维度相同）
            use_bias: 是否使用偏置
        """
        self.embed_dim = embed_dim
        self.use_bias = use_bias

        # 初始化Q、K、V的投影矩阵
        # Q用于投影解码器输入，K、V用于投影编码器输出
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        if use_bias:
            self.b_q = np.zeros(embed_dim)
            self.b_k = np.zeros(embed_dim)
            self.b_v = np.zeros(embed_dim)

    def forward(self, decoder_input, encoder_output, mask=None):
        """
        前向传播

        Args:
            decoder_input: 解码器输入（目标序列），形状为 (tgt_len, embed_dim)
            encoder_output: 编码器输出（源序列），形状为 (src_len, embed_dim)
            mask: 注意力掩码，形状为 (tgt_len, src_len)
                  用于屏蔽某些位置（如padding）

        Returns:
            output: 输出序列，形状为 (tgt_len, embed_dim)
            attention_weights: 注意力权重，形状为 (tgt_len, src_len)

        注意：
        - tgt_len: 目标序列长度（解码器）
        - src_len: 源序列长度（编码器）
        - 两者可以不同！
        """
        tgt_len, embed_dim = decoder_input.shape
        src_len = encoder_output.shape[0]

        # 步骤1: 线性投影得到Q、K、V
        # Q来自解码器，K和V来自编码器
        Q = np.dot(decoder_input, self.W_q)     # (tgt_len, embed_dim)
        K = np.dot(encoder_output, self.W_k)    # (src_len, embed_dim)
        V = np.dot(encoder_output, self.W_v)    # (src_len, embed_dim)

        if self.use_bias:
            Q += self.b_q
            K += self.b_k
            V += self.b_v

        # 步骤2: 计算注意力得分（Scaled Dot-Product）
        # Q @ K^T: (tgt_len, embed_dim) @ (embed_dim, src_len) = (tgt_len, src_len)
        # 解码器的每个位置关注编码器的所有位置
        scores = np.dot(Q, K.T) / np.sqrt(embed_dim)  # (tgt_len, src_len)

        # 步骤3: 应用mask（如果提供）
        if mask is not None:
            # mask中为0的位置设置为负无穷，softmax后会变成0
            scores = np.where(mask == 0, -1e9, scores)

        # 步骤4: Softmax归一化得到注意力权重
        # 对最后一维（src维度）做softmax
        attention_weights = softmax(scores, axis=-1)  # (tgt_len, src_len)

        # 步骤5: 加权求和Value
        # (tgt_len, src_len) @ (src_len, embed_dim) = (tgt_len, embed_dim)
        output = np.dot(attention_weights, V)  # (tgt_len, embed_dim)

        return output, attention_weights


class EncoderDecoderWithCrossAttention:
    """
    简化的Encoder-Decoder模型（仅包含Cross-Attention）

    用于演示Cross-Attention在Seq2Seq任务中的作用
    """

    def __init__(self, embed_dim):
        """
        初始化Encoder-Decoder模型

        Args:
            embed_dim: 嵌入维度
        """
        self.embed_dim = embed_dim
        self.cross_attention = CrossAttention(embed_dim)

    def forward(self, src_seq, tgt_seq, src_padding_mask=None):
        """
        前向传播

        Args:
            src_seq: 源序列（已编码），形状为 (src_len, embed_dim)
            tgt_seq: 目标序列（已编码），形状为 (tgt_len, embed_dim)
            src_padding_mask: 源序列padding mask，形状为 (tgt_len, src_len)

        Returns:
            output: 解码器输出，形状为 (tgt_len, embed_dim)
            attention_weights: 交叉注意力权重，形状为 (tgt_len, src_len)
        """
        # 在实际的Transformer中，src_seq会先经过Encoder处理
        # tgt_seq会先经过Decoder的Self-Attention处理
        # 这里简化处理，直接使用Cross-Attention

        encoder_output = src_seq
        decoder_hidden = tgt_seq

        # Cross-Attention: 解码器关注编码器
        output, attention_weights = self.cross_attention.forward(
            decoder_hidden,
            encoder_output,
            mask=src_padding_mask
        )

        return output, attention_weights


def create_padding_mask(src_len, tgt_len, valid_src_len):
    """
    创建padding mask

    Args:
        src_len: 源序列总长度（包括padding）
        tgt_len: 目标序列长度
        valid_src_len: 源序列的有效长度（不包括padding）

    Returns:
        mask: padding mask，形状为 (tgt_len, src_len)
              有效位置为1，padding位置为0
    """
    mask = np.zeros((tgt_len, src_len))
    mask[:, :valid_src_len] = 1
    return mask


def cross_attention_simple(decoder_input, encoder_output):
    """
    简化版交叉注意力（不使用可学习参数）

    Args:
        decoder_input: 解码器输入，形状为 (tgt_len, embed_dim)
        encoder_output: 编码器输出，形状为 (src_len, embed_dim)

    Returns:
        output: 输出序列，形状为 (tgt_len, embed_dim)
        weights: 注意力权重，形状为 (tgt_len, src_len)
    """
    tgt_len, embed_dim = decoder_input.shape

    # 直接使用输入作为Q、K、V
    Q = decoder_input        # (tgt_len, embed_dim)
    K = V = encoder_output   # (src_len, embed_dim)

    # 计算注意力得分
    scores = np.dot(Q, K.T) / np.sqrt(embed_dim)  # (tgt_len, src_len)

    # Softmax
    weights = softmax(scores, axis=-1)  # (tgt_len, src_len)

    # 加权求和
    output = np.dot(weights, V)  # (tgt_len, embed_dim)

    return output, weights


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("交叉注意力机制演示")
    print("=" * 70)

    # 参数设置
    embed_dim = 64
    src_len = 10  # 源序列长度（如英文句子）
    tgt_len = 8   # 目标序列长度（如中文句子）

    # 生成示例输入
    # 编码器输出（源序列的表示）
    encoder_output = np.random.randn(src_len, embed_dim)
    # 解码器输入（目标序列的表示）
    decoder_input = np.random.randn(tgt_len, embed_dim)

    # 创建交叉注意力层
    cross_attn = CrossAttention(embed_dim)

    # 前向传播（无mask）
    print("\n1. 标准交叉注意力（无mask）")
    output, attn_weights = cross_attn.forward(decoder_input, encoder_output)

    print(f"编码器输出形状（源序列）: {encoder_output.shape}")
    print(f"解码器输入形状（目标序列）: {decoder_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"\n注意力权重矩阵（前3行，目标序列的前3个位置）:")
    print(attn_weights[:3, :])
    print(f"\n每行权重和（应该都是1.0）: {np.sum(attn_weights, axis=1)[:3]}")

    # 使用padding mask
    print("\n" + "=" * 70)
    print("2. 带Padding Mask的交叉注意力")
    valid_src_len = 7  # 假设源序列实际长度为7，后3个是padding
    padding_mask = create_padding_mask(src_len, tgt_len, valid_src_len)
    print(f"Padding Mask（1表示有效，0表示padding）:")
    print(f"形状: {padding_mask.shape}")
    print(f"前2行:\n{padding_mask[:2, :].astype(int)}")

    output_masked, attn_weights_masked = cross_attn.forward(
        decoder_input, encoder_output, mask=padding_mask
    )
    print(f"\n带mask的注意力权重（前2行）:")
    print(attn_weights_masked[:2, :])
    print("注意：后3个位置（padding）的权重为0")

    # 机器翻译场景模拟
    print("\n" + "=" * 70)
    print("3. 机器翻译场景模拟")
    print("=" * 70)

    # 模拟：英文 "I love machine learning" -> 中文 "我爱机器学习"
    src_words = ["I", "love", "machine", "learning", "<PAD>", "<PAD>"]
    tgt_words = ["我", "爱", "机器", "学习"]

    src_len_sim = len(src_words)
    tgt_len_sim = len(tgt_words)
    valid_src_len_sim = 4  # 实际有效的源词数

    # 创建词嵌入（简化处理）
    encoder_output_sim = np.random.randn(src_len_sim, embed_dim)
    decoder_input_sim = np.random.randn(tgt_len_sim, embed_dim)

    # 创建padding mask
    padding_mask_sim = create_padding_mask(src_len_sim, tgt_len_sim, valid_src_len_sim)

    # 交叉注意力
    output_sim, attn_weights_sim = cross_attn.forward(
        decoder_input_sim, encoder_output_sim, mask=padding_mask_sim
    )

    print(f"源语言（英文）: {' '.join(src_words)}")
    print(f"目标语言（中文）: {' '.join(tgt_words)}")
    print(f"\n交叉注意力权重矩阵 (目标词 × 源词):")
    print("       ", "  ".join(f"{w:>8}" for w in src_words))
    for i, tgt_word in enumerate(tgt_words):
        weights_str = "  ".join(f"{w:>8.3f}" for w in attn_weights_sim[i])
        print(f"{tgt_word:>4}: {weights_str}")

    print(f"\n每个目标词最关注的源词:")
    for i, tgt_word in enumerate(tgt_words):
        max_idx = np.argmax(attn_weights_sim[i])
        max_weight = attn_weights_sim[i, max_idx]
        print(f"  '{tgt_word}' -> '{src_words[max_idx]}' (权重: {max_weight:.3f})")

    # 完整的Encoder-Decoder示例
    print("\n" + "=" * 70)
    print("4. Encoder-Decoder模型示例")
    model = EncoderDecoderWithCrossAttention(embed_dim)
    output_model, attn_model = model.forward(
        encoder_output_sim, decoder_input_sim, padding_mask_sim
    )
    print(f"Encoder-Decoder输出形状: {output_model.shape}")
    print(f"注意力权重形状: {attn_model.shape}")

    # 简化版交叉注意力
    print("\n" + "=" * 70)
    print("5. 简化版交叉注意力（无参数）")
    output_simple, weights_simple = cross_attention_simple(decoder_input, encoder_output)
    print(f"输出形状: {output_simple.shape}")
    print(f"注意力权重形状: {weights_simple.shape}")

    print("\n" + "=" * 70)
    print("交叉注意力 vs 自注意力的关键区别:")
    print("=" * 70)
    print("【自注意力 (Self-Attention)】")
    print("  • Query、Key、Value都来自同一个序列")
    print("  • 注意力矩阵是方阵 (seq_len × seq_len)")
    print("  • 用于捕获序列内部的依赖关系")
    print("  • 应用：BERT、GPT的内部表示")
    print()
    print("【交叉注意力 (Cross-Attention)】")
    print("  • Query来自解码器（目标序列）")
    print("  • Key、Value来自编码器（源序列）")
    print("  • 注意力矩阵不一定是方阵 (tgt_len × src_len)")
    print("  • 用于连接两个不同的序列")
    print("  • 应用：机器翻译、图像描述生成")
    print()
    print("【典型应用场景】")
    print("  ✓ 机器翻译: 目标语言关注源语言")
    print("  ✓ 图像描述: 文本关注图像区域")
    print("  ✓ 语音识别: 文本关注音频帧")
    print("  ✓ 问答系统: 答案关注问题和上下文")
