"""
Transformer Decoder 实现

Decoder是Transformer架构的核心组件，用于自回归生成任务（如GPT）。
包含Self-Attention、Cross-Attention（可选）和Feed-Forward Network。

核心组件：
1. Masked Self-Attention: 只能看到当前位置之前的token（因果性）
2. Cross-Attention: 关注Encoder的输出（用于Encoder-Decoder架构）
3. Feed-Forward Network: 位置独立的全连接层
4. Layer Normalization: 稳定训练
5. Residual Connection: 梯度流动

架构类型：
1. Decoder-only (GPT): 只有Masked Self-Attention + FFN
2. Encoder-Decoder (BERT/T5): 包含Cross-Attention

应用：GPT系列、LLaMA、PaLM等所有自回归大语言模型
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """
    GELU激活函数（Gaussian Error Linear Unit）

    GPT使用的激活函数，比ReLU更平滑。
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class LayerNorm:
    """层归一化"""

    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class MultiHeadSelfAttention:
    """
    多头自注意力（带Causal Mask）

    用于Decoder，只能看到当前位置之前的token。
    """

    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q、K、V投影
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # 输出投影
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def split_heads(self, x):
        """分割成多个头"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)

    def combine_heads(self, x):
        """合并多个头"""
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.embed_dim)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入，形状为 (seq_len, embed_dim)
            mask: Causal mask，形状为 (seq_len, seq_len)

        Returns:
            output: 输出，形状为 (seq_len, embed_dim)
        """
        seq_len = x.shape[0]

        # 投影到Q、K、V
        Q = np.dot(x, self.W_q.T)
        K = np.dot(x, self.W_k.T)
        V = np.dot(x, self.W_v.T)

        # 分割成多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算每个头的attention
        outputs = []
        for i in range(self.num_heads):
            Q_i = Q[i]
            K_i = K[i]
            V_i = V[i]

            # Scaled dot-product attention
            scores = np.dot(Q_i, K_i.T) / np.sqrt(self.head_dim)

            # 应用causal mask
            if mask is not None:
                scores = np.where(mask == 0, -1e9, scores)

            attn_weights = softmax(scores, axis=-1)
            output_i = np.dot(attn_weights, V_i)

            outputs.append(output_i)

        # 合并所有头
        multi_head_output = np.stack(outputs, axis=0)
        concatenated = self.combine_heads(multi_head_output)

        # 输出投影
        output = np.dot(concatenated, self.W_o.T)

        return output


class MultiHeadCrossAttention:
    """
    多头交叉注意力

    Query来自Decoder，Key和Value来自Encoder输出。
    用于Encoder-Decoder架构（如T5、BART）。
    """

    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def split_heads(self, x):
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 0, 2)

    def combine_heads(self, x):
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.embed_dim)

    def forward(self, query, key_value):
        """
        前向传播

        Args:
            query: Decoder的输入，形状为 (tgt_len, embed_dim)
            key_value: Encoder的输出，形状为 (src_len, embed_dim)

        Returns:
            output: 输出，形状为 (tgt_len, embed_dim)
        """
        # Query来自Decoder
        Q = np.dot(query, self.W_q.T)

        # Key和Value来自Encoder
        K = np.dot(key_value, self.W_k.T)
        V = np.dot(key_value, self.W_v.T)

        # 分割成多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算attention
        outputs = []
        for i in range(self.num_heads):
            Q_i = Q[i]
            K_i = K[i]
            V_i = V[i]

            scores = np.dot(Q_i, K_i.T) / np.sqrt(self.head_dim)
            attn_weights = softmax(scores, axis=-1)
            output_i = np.dot(attn_weights, V_i)

            outputs.append(output_i)

        # 合并头
        multi_head_output = np.stack(outputs, axis=0)
        concatenated = self.combine_heads(multi_head_output)

        # 输出投影
        output = np.dot(concatenated, self.W_o.T)

        return output


class FeedForwardNetwork:
    """
    前馈神经网络（Position-wise Feed-Forward Network）

    两层全连接网络，对每个位置独立应用。
    标准配置：d_model → 4*d_model → d_model
    """

    def __init__(self, embed_dim, ffn_dim=None, activation='gelu'):
        if ffn_dim is None:
            ffn_dim = 4 * embed_dim  # 标准配置

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim

        # 两层全连接
        self.W1 = np.random.randn(ffn_dim, embed_dim) / np.sqrt(embed_dim)
        self.b1 = np.zeros(ffn_dim)

        self.W2 = np.random.randn(embed_dim, ffn_dim) / np.sqrt(ffn_dim)
        self.b2 = np.zeros(embed_dim)

        # 激活函数
        self.activation = gelu if activation == 'gelu' else lambda x: np.maximum(0, x)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入，形状为 (seq_len, embed_dim)

        Returns:
            output: 输出，形状为 (seq_len, embed_dim)
        """
        # 第一层：embed_dim -> ffn_dim
        hidden = np.dot(x, self.W1.T) + self.b1
        hidden = self.activation(hidden)

        # 第二层：ffn_dim -> embed_dim
        output = np.dot(hidden, self.W2.T) + self.b2

        return output


class TransformerDecoderLayer:
    """
    Transformer Decoder层

    包含：
    1. Masked Self-Attention + LayerNorm + Residual
    2. Cross-Attention + LayerNorm + Residual（可选，用于Encoder-Decoder）
    3. FFN + LayerNorm + Residual
    """

    def __init__(self, embed_dim, num_heads, ffn_dim=None, has_cross_attention=False):
        self.embed_dim = embed_dim
        self.has_cross_attention = has_cross_attention

        # 1. Masked Self-Attention
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)

        # 2. Cross-Attention（可选）
        if has_cross_attention:
            self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)
            self.ln2 = LayerNorm(embed_dim)

        # 3. Feed-Forward Network
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.ln3 = LayerNorm(embed_dim)

    def forward(self, x, encoder_output=None, causal_mask=None):
        """
        前向传播

        Args:
            x: Decoder输入，形状为 (seq_len, embed_dim)
            encoder_output: Encoder输出（用于Cross-Attention），形状为 (src_len, embed_dim)
            causal_mask: 因果mask，形状为 (seq_len, seq_len)

        Returns:
            output: 输出，形状为 (seq_len, embed_dim)
        """
        # 1. Masked Self-Attention + Residual + LayerNorm
        attn_output = self.self_attn.forward(x, mask=causal_mask)
        x = self.ln1.forward(x + attn_output)  # Post-LN

        # 2. Cross-Attention + Residual + LayerNorm（如果有）
        if self.has_cross_attention and encoder_output is not None:
            cross_output = self.cross_attn.forward(x, encoder_output)
            x = self.ln2.forward(x + cross_output)

        # 3. FFN + Residual + LayerNorm
        ffn_output = self.ffn.forward(x)
        x = self.ln3.forward(x + ffn_output)

        return x


class TransformerDecoder:
    """
    完整的Transformer Decoder

    堆叠多层DecoderLayer，用于自回归生成。

    两种模式：
    1. Decoder-only (GPT): 只有Self-Attention
    2. Encoder-Decoder (T5/BART): 包含Cross-Attention
    """

    def __init__(self, num_layers, embed_dim, num_heads, ffn_dim=None,
                 has_cross_attention=False, vocab_size=None):
        """
        初始化Transformer Decoder

        Args:
            num_layers: Decoder层数
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            ffn_dim: FFN中间层维度（默认4*embed_dim）
            has_cross_attention: 是否包含Cross-Attention
            vocab_size: 词表大小（用于输出层）
        """
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # 堆叠多层Decoder
        self.layers = [
            TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, has_cross_attention)
            for _ in range(num_layers)
        ]

        # 最终LayerNorm
        self.final_ln = LayerNorm(embed_dim)

        # 输出层（Language Model Head）
        if vocab_size is not None:
            self.lm_head = np.random.randn(vocab_size, embed_dim) / np.sqrt(embed_dim)
        else:
            self.lm_head = None

    def forward(self, x, encoder_output=None, causal_mask=None, return_logits=False):
        """
        前向传播

        Args:
            x: 输入，形状为 (seq_len, embed_dim)
            encoder_output: Encoder输出（用于Cross-Attention）
            causal_mask: 因果mask
            return_logits: 是否返回logits（需要lm_head）

        Returns:
            output: 输出，形状为 (seq_len, embed_dim) 或 (seq_len, vocab_size)
        """
        # 逐层处理
        for layer in self.layers:
            x = layer.forward(x, encoder_output, causal_mask)

        # 最终LayerNorm
        x = self.final_ln.forward(x)

        # 如果需要输出logits
        if return_logits and self.lm_head is not None:
            logits = np.dot(x, self.lm_head.T)  # (seq_len, vocab_size)
            return logits

        return x


def create_causal_mask(seq_len):
    """
    创建因果mask（下三角矩阵）

    Args:
        seq_len: 序列长度

    Returns:
        mask: 形状为 (seq_len, seq_len) 的下三角矩阵
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("Transformer Decoder演示")
    print("=" * 70)

    # ========== 1. 单层Decoder Layer ==========
    print("\n" + "=" * 70)
    print("1. 单层Decoder Layer（Decoder-only风格）")
    print("=" * 70)

    embed_dim = 512
    num_heads = 8
    seq_len = 10

    # 创建单层
    decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, has_cross_attention=False)

    # 输入
    x = np.random.randn(seq_len, embed_dim)

    # 创建因果mask
    causal_mask = create_causal_mask(seq_len)

    print(f"\n配置:")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  序列长度: {seq_len}")

    print(f"\nCausal Mask (前5×5):")
    print(causal_mask[:5, :5].astype(int))

    # 前向传播
    output = decoder_layer.forward(x, causal_mask=causal_mask)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # ========== 2. Encoder-Decoder风格（带Cross-Attention）==========
    print("\n" + "=" * 70)
    print("2. 单层Decoder Layer（Encoder-Decoder风格）")
    print("=" * 70)

    decoder_layer_enc_dec = TransformerDecoderLayer(
        embed_dim, num_heads, has_cross_attention=True
    )

    # Encoder输出（模拟）
    src_len = 15
    encoder_output = np.random.randn(src_len, embed_dim)

    print(f"\n配置:")
    print(f"  Decoder输入长度: {seq_len}")
    print(f"  Encoder输出长度: {src_len}")
    print(f"  包含Cross-Attention: True")

    # 前向传播
    output_enc_dec = decoder_layer_enc_dec.forward(
        x, encoder_output=encoder_output, causal_mask=causal_mask
    )

    print(f"\nDecoder输入: {x.shape}")
    print(f"Encoder输出: {encoder_output.shape}")
    print(f"Decoder输出: {output_enc_dec.shape}")

    # ========== 3. 完整Decoder（GPT风格）==========
    print("\n" + "=" * 70)
    print("3. 完整Transformer Decoder（GPT风格）")
    print("=" * 70)

    num_layers = 12
    vocab_size = 50000

    # 创建GPT风格的Decoder
    gpt_decoder = TransformerDecoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        has_cross_attention=False,  # GPT只有Self-Attention
        vocab_size=vocab_size
    )

    print(f"\n配置（类似GPT）:")
    print(f"  层数: {num_layers}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  词表大小: {vocab_size}")

    # 前向传播
    output_gpt = gpt_decoder.forward(x, causal_mask=causal_mask, return_logits=False)
    logits_gpt = gpt_decoder.forward(x, causal_mask=causal_mask, return_logits=True)

    print(f"\n输入: {x.shape}")
    print(f"Hidden输出: {output_gpt.shape}")
    print(f"Logits输出: {logits_gpt.shape}")

    # ========== 4. 完整Decoder（T5风格）==========
    print("\n" + "=" * 70)
    print("4. 完整Transformer Decoder（T5/BART风格）")
    print("=" * 70)

    # 创建T5风格的Decoder（带Cross-Attention）
    t5_decoder = TransformerDecoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        has_cross_attention=True,  # T5有Cross-Attention
        vocab_size=vocab_size
    )

    print(f"\n配置（类似T5）:")
    print(f"  层数: {num_layers}")
    print(f"  包含Cross-Attention: True")

    # 前向传播（需要encoder_output）
    output_t5 = t5_decoder.forward(
        x,
        encoder_output=encoder_output,
        causal_mask=causal_mask,
        return_logits=False
    )

    print(f"\nDecoder输入: {x.shape}")
    print(f"Encoder输出: {encoder_output.shape}")
    print(f"Decoder输出: {output_t5.shape}")

    # ========== 5. 参数量分析 ==========
    print("\n" + "=" * 70)
    print("5. 参数量分析")
    print("=" * 70)

    def count_params(embed_dim, num_heads, num_layers, vocab_size, has_cross_attn):
        """估算参数量"""
        # 每层的参数
        # Self-Attention: 4 * embed_dim^2 (Q,K,V,O)
        # Cross-Attention: 4 * embed_dim^2 (如果有)
        # FFN: 2 * embed_dim * (4*embed_dim) = 8 * embed_dim^2
        # LayerNorm: 2 * embed_dim (γ和β) × 层数中的LN数量

        params_per_layer = 0

        # Self-Attention
        params_per_layer += 4 * embed_dim * embed_dim

        # Cross-Attention（如果有）
        if has_cross_attn:
            params_per_layer += 4 * embed_dim * embed_dim

        # FFN
        params_per_layer += 2 * embed_dim * (4 * embed_dim)

        # LayerNorm（每层有2-3个）
        ln_per_layer = 3 if has_cross_attn else 2
        params_per_layer += ln_per_layer * 2 * embed_dim

        # 总参数
        total_params = num_layers * params_per_layer

        # 最终LayerNorm
        total_params += 2 * embed_dim

        # LM Head
        total_params += vocab_size * embed_dim

        return total_params, params_per_layer

    # GPT风格
    gpt_total, gpt_per_layer = count_params(embed_dim, num_heads, num_layers, vocab_size, False)

    # T5风格
    t5_total, t5_per_layer = count_params(embed_dim, num_heads, num_layers, vocab_size, True)

    print(f"\nGPT风格（Decoder-only）:")
    print(f"  每层参数: {gpt_per_layer:,}")
    print(f"  总参数: {gpt_total:,}")
    print(f"  约 {gpt_total / 1e6:.1f}M 参数")

    print(f"\nT5风格（Encoder-Decoder）:")
    print(f"  每层参数: {t5_per_layer:,}")
    print(f"  总参数: {t5_total:,}")
    print(f"  约 {t5_total / 1e6:.1f}M 参数")

    # ========== 6. 实际模型规模 ==========
    print("\n" + "=" * 70)
    print("6. 实际模型规模对比")
    print("=" * 70)

    models = [
        ("GPT-2 Small", 12, 768, 12, 50257, False),
        ("GPT-2 Medium", 24, 1024, 16, 50257, False),
        ("GPT-2 Large", 36, 1280, 20, 50257, False),
        ("GPT-3", 96, 12288, 96, 50257, False),
        ("LLaMA-7B", 32, 4096, 32, 32000, False),
    ]

    print(f"\n{'模型':<15} {'层数':<6} {'维度':<6} {'头数':<6} {'参数量':<15}")
    print("-" * 60)

    for name, layers, dim, heads, vocab, cross in models:
        total, _ = count_params(dim, heads, layers, vocab, cross)
        print(f"{name:<15} {layers:<6} {dim:<6} {heads:<6} {total / 1e9:.1f}B")

    # ========== 7. Decoder的关键特性 ==========
    print("\n" + "=" * 70)
    print("Transformer Decoder的关键特性:")
    print("=" * 70)
    print("✓ Masked Self-Attention: 保证因果性（自回归）")
    print("✓ Cross-Attention: 连接Encoder和Decoder（可选）")
    print("✓ Position-wise FFN: 对每个位置独立处理")
    print("✓ Residual Connection: 改善梯度流动")
    print("✓ Layer Normalization: 稳定训练")
    print("✓ 两种架构: Decoder-only (GPT) vs Encoder-Decoder (T5)")
    print("✓ 应用: 所有自回归语言模型（GPT、LLaMA、Claude等）")
