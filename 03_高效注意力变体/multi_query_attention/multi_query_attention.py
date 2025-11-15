"""
Multi-Query Attention (MQA，多查询注意力) 实现

MQA是一种极致高效的注意力机制，由Noam Shazeer在2019年的论文
"Fast Transformer Decoding: One Write-Head is All You Need"中提出。

核心思想：
1. 所有查询头共享同一对键值头
2. 最大化减少KV缓存（相比MHA减少h倍）
3. 显著提升推理速度，特别是自回归生成
4. 轻微牺牲模型质量换取效率

公式：
对于所有查询头i：
  head_i = Attention(Q_i W_i^Q, K W^K, V W^V)

其中所有头共享同一个K和V

参数对比（embed_dim=d, num_heads=h）：
- MHA: Q,K,V各需要 h 个头 → KV缓存: 2 × h × d_k × seq_len
- GQA: Q需要 h 个头，K,V各需要 h/G 个头 → KV缓存: 2 × (h/G) × d_k × seq_len
- MQA: Q需要 h 个头，K,V各需要 1 个头 → KV缓存: 2 × 1 × d_k × seq_len

应用：PaLM、Falcon、StarCoder等模型
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


class MultiQueryAttention:
    """
    Multi-Query Attention (多查询注意力)

    特点：
    - 所有查询头共享单一的键值头
    - 最大化减少KV缓存（相比MHA减少num_heads倍）
    - 推理速度最快，内存占用最小
    - 可能轻微降低模型质量（相比MHA/GQA）

    参数量对比（以embed_dim=512, num_heads=8为例）：
    - MHA: Q,K,V投影都是512×512，KV有8个头
    - MQA: Q投影512×512，K,V投影512×64，KV只有1个头

    MQA vs GQA:
    - MQA是GQA的特例（num_kv_heads=1）
    - MQA效率最高，但可能损失质量
    - GQA在两者间取得平衡
    """

    def __init__(self, embed_dim, num_heads, use_bias=True):
        """
        初始化多查询注意力层

        Args:
            embed_dim: 嵌入维度（必须能被num_heads整除）
            num_heads: 查询头的数量
            use_bias: 是否使用偏置
        """
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_bias = use_bias

        # Q投影矩阵：完整的num_heads个头
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # K,V投影矩阵：只有1个头（关键！）
        # 这是MQA与MHA/GQA的核心区别
        self.W_k = np.random.randn(embed_dim, self.head_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, self.head_dim) / np.sqrt(embed_dim)

        # 输出投影矩阵
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        if use_bias:
            self.b_q = np.zeros(embed_dim)
            self.b_k = np.zeros(self.head_dim)
            self.b_v = np.zeros(self.head_dim)
            self.b_o = np.zeros(embed_dim)

    def split_heads(self, x):
        """
        将查询分割成多个头

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
            attention_weights: (可选) 所有头的注意力权重
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
        K = np.dot(key, self.W_k)    # (seq_len_k, head_dim) - 注意：只有1个头！
        V = np.dot(value, self.W_v)  # (seq_len_v, head_dim) - 注意：只有1个头！

        if self.use_bias:
            Q += self.b_q
            K += self.b_k
            V += self.b_v

        # 步骤2: 将Q分割成多个头，K和V保持单头
        Q = self.split_heads(Q)  # (num_heads, seq_len_q, head_dim)
        # K和V已经是单头，不需要分割
        # K: (seq_len_k, head_dim)
        # V: (seq_len_v, head_dim)

        # 步骤3: MQA的核心 - 所有查询头共享同一对KV头
        head_outputs = []
        attention_weights_list = []

        for i in range(self.num_heads):
            # 取出第i个查询头
            Q_i = Q[i]  # (seq_len_q, head_dim)

            # 关键：所有查询头都使用同一个K和V！
            # 这是MQA与MHA/GQA的核心区别
            head_output, attn_weights = scaled_dot_product_attention(
                Q_i, K, V, mask=mask  # 注意：K和V对所有头都相同
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
            attention_weights = np.stack(attention_weights_list, axis=0)
            return output, attention_weights

        return output

    def get_num_parameters(self):
        """
        计算参数量

        Returns:
            total_params: 总参数量
            param_breakdown: 参数详细分解
        """
        params = {
            'W_q': self.embed_dim * self.embed_dim,
            'W_k': self.embed_dim * self.head_dim,  # 只有1个头
            'W_v': self.embed_dim * self.head_dim,  # 只有1个头
            'W_o': self.embed_dim * self.embed_dim,
        }

        if self.use_bias:
            params['b_q'] = self.embed_dim
            params['b_k'] = self.head_dim
            params['b_v'] = self.head_dim
            params['b_o'] = self.embed_dim

        total = sum(params.values())
        return total, params


class MultiQuerySelfAttention(MultiQueryAttention):
    """
    Multi-Query Self-Attention (多查询自注意力)

    是MultiQueryAttention的特例，Q、K、V都来自同一个输入。
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


def compare_attention_variants(embed_dim, num_heads, seq_len):
    """
    比较MHA、GQA、MQA的参数量和KV缓存

    Args:
        embed_dim: 嵌入维度
        num_heads: 头数
        seq_len: 序列长度
    """
    head_dim = embed_dim // num_heads

    print("=" * 80)
    print("MHA vs GQA vs MQA 对比分析")
    print("=" * 80)
    print(f"配置: embed_dim={embed_dim}, num_heads={num_heads}, seq_len={seq_len}")
    print(f"每个头维度: {head_dim}\n")

    # 计算参数量
    # MHA
    mha_params = 4 * embed_dim * embed_dim
    mha_kv_cache = 2 * seq_len * num_heads * head_dim

    # GQA (假设4个KV头)
    if num_heads >= 4:
        num_kv_heads_gqa = num_heads // 4
        gqa_params = (2 * embed_dim * embed_dim +  # W_q, W_o
                     2 * embed_dim * num_kv_heads_gqa * head_dim)  # W_k, W_v
        gqa_kv_cache = 2 * seq_len * num_kv_heads_gqa * head_dim
    else:
        num_kv_heads_gqa = 1
        gqa_params = mha_params
        gqa_kv_cache = mha_kv_cache

    # MQA
    mqa_params = (2 * embed_dim * embed_dim +  # W_q, W_o
                 2 * embed_dim * head_dim)      # W_k, W_v (只有1个头)
    mqa_kv_cache = 2 * seq_len * 1 * head_dim

    # 打印对比表
    print(f"{'变体':<15} {'Q头数':>8} {'KV头数':>8} {'参数量':>15} {'KV缓存':>15} {'缓存节省':>10}")
    print("-" * 80)
    print(f"{'MHA':<15} {num_heads:>8} {num_heads:>8} {mha_params:>15,} {mha_kv_cache:>15,} {0:>9.1%}")
    print(f"{'GQA':<15} {num_heads:>8} {num_kv_heads_gqa:>8} {gqa_params:>15,} {gqa_kv_cache:>15,} "
          f"{(1 - gqa_kv_cache/mha_kv_cache):>9.1%}")
    print(f"{'MQA':<15} {num_heads:>8} {1:>8} {mqa_params:>15,} {mqa_kv_cache:>15,} "
          f"{(1 - mqa_kv_cache/mha_kv_cache):>9.1%}")

    print("\n关键观察:")
    print(f"  ✓ MQA相比MHA节省KV缓存: {(1 - mqa_kv_cache/mha_kv_cache)*100:.1f}%")
    print(f"  ✓ MQA相比MHA减少参数: {(mha_params - mqa_params):,} (-{(1-mqa_params/mha_params)*100:.1f}%)")
    print(f"  ✓ MQA是最极致的KV缓存优化方案")
    print(f"  ✓ 适合对推理速度要求极高的场景")


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("Multi-Query Attention (多查询注意力) 演示")
    print("=" * 80)

    # 参数设置
    seq_len = 10
    embed_dim = 512
    num_heads = 8

    print(f"\n配置:")
    print(f"  序列长度: {seq_len}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  查询头数(Q): {num_heads}")
    print(f"  键值头数(K,V): 1 ← MQA的核心特点！")
    print(f"  每个头的维度: {embed_dim // num_heads}")

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 创建MQA层
    mqa = MultiQuerySelfAttention(embed_dim, num_heads)

    # 前向传播
    print("\n" + "=" * 80)
    print("1. 标准MQA（无mask）")
    print("=" * 80)
    output, attn_weights = mqa.forward(x, return_attention=True)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"  - {num_heads}个查询头")
    print(f"  - 所有头共享同一对KV头（MQA的核心）")

    # 参数量分析
    print("\n" + "=" * 80)
    print("2. 参数量分析")
    print("=" * 80)

    total_params, param_breakdown = mqa.get_num_parameters()

    print(f"\nMQA参数详情:")
    for name, count in param_breakdown.items():
        print(f"  {name}: {count:,}")
    print(f"  总计: {total_params:,}")

    print(f"\n与MHA对比:")
    mha_params = 4 * embed_dim * embed_dim
    print(f"  MHA参数量: {mha_params:,}")
    print(f"  MQA参数量: {total_params:,}")
    print(f"  减少: {mha_params - total_params:,} (-{(1-total_params/mha_params)*100:.1f}%)")

    # 对比MHA、GQA、MQA
    print("\n" + "=" * 80)
    print("3. 对比MHA、GQA、MQA")
    print("=" * 80)
    compare_attention_variants(embed_dim, num_heads, seq_len)

    # KV头共享效果
    print("\n" + "=" * 80)
    print("4. KV头共享效果")
    print("=" * 80)

    print("\n所有查询头使用相同的K和V:")
    print(f"  查询头0-{num_heads-1} 都共享 KV头0")
    print("\n这意味着:")
    print("  ✓ KV缓存是MHA的1/8")
    print("  ✓ 推理时内存占用大幅降低")
    print("  ✓ 但所有查询头看到相同的K和V")
    print("  ✓ 只能通过不同的Q投影来产生多样性")

    # 因果mask示例
    print("\n" + "=" * 80)
    print("5. 带因果mask的MQA（自回归生成）")
    print("=" * 80)

    causal_mask = create_causal_mask(seq_len)
    output_masked, attn_weights_masked = mqa.forward(x, mask=causal_mask, return_attention=True)

    print(f"因果mask形状: {causal_mask.shape}")
    print(f"带mask的输出形状: {output_masked.shape}")

    # 显示不同头的注意力模式
    print(f"\n不同查询头的注意力权重（位置0）:")
    for i in range(min(4, num_heads)):
        print(f"  头{i}: {attn_weights_masked[i, 0, :5]}")
    print("注意：虽然共享KV，但不同Q头仍能学到不同的注意力模式")

    # 实际应用示例
    print("\n" + "=" * 80)
    print("6. 实际应用：PaLM配置")
    print("=" * 80)

    # PaLM使用MQA
    palm_embed_dim = 8192
    palm_num_heads = 32
    palm_seq_len = 2048

    palm_mqa = MultiQuerySelfAttention(palm_embed_dim, palm_num_heads)
    palm_params, _ = palm_mqa.get_num_parameters()

    print(f"\nPaLM 540B 配置:")
    print(f"  嵌入维度: {palm_embed_dim}")
    print(f"  查询头数: {palm_num_heads}")
    print(f"  键值头数: 1 (MQA)")
    print(f"  序列长度: {palm_seq_len}")
    print(f"  单层注意力参数量: {palm_params:,}")

    # KV缓存对比
    head_dim = palm_embed_dim // palm_num_heads
    mha_kv_cache = 2 * palm_seq_len * palm_num_heads * head_dim
    mqa_kv_cache = 2 * palm_seq_len * 1 * head_dim

    print(f"\nKV缓存对比（序列长度={palm_seq_len}）:")
    print(f"  如果使用MHA: {mha_kv_cache:,} 个浮点数 ({mha_kv_cache * 4 / 1024 / 1024:.1f} MB)")
    print(f"  实际MQA缓存: {mqa_kv_cache:,} 个浮点数 ({mqa_kv_cache * 4 / 1024 / 1024:.1f} MB)")
    print(f"  节省: {(mha_kv_cache - mqa_kv_cache) * 4 / 1024 / 1024:.1f} MB ({(1 - mqa_kv_cache/mha_kv_cache)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("MQA的关键优势:")
    print("=" * 80)
    print("✓ 最大化减少KV缓存（相比MHA减少num_heads倍）")
    print("✓ 推理速度最快，内存占用最小")
    print("✓ 参数量显著减少")
    print("✓ 特别适合超大规模模型和长序列")
    print("✓ 被PaLM、Falcon、StarCoder等模型采用")
    print("\n注意事项:")
    print("⚠ 可能轻微降低模型质量（相比MHA/GQA）")
    print("⚠ 所有查询头共享KV可能限制表达能力")
    print("⚠ 需要权衡质量和效率")
