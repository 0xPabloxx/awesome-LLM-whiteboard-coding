"""
Grouped-Query Attention (GQA，分组查询注意力) 实现

GQA是介于Multi-Head Attention (MHA)和Multi-Query Attention (MQA)之间的高效注意力机制。
由Ainslie等人在2023年的论文"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"中提出。

核心思想：
1. 将查询头分成G组
2. 每组共享一对键值头
3. 平衡了MHA的质量和MQA的效率
4. 显著减少KV缓存，提升推理速度

公式：
对于每个组g：
  head_i = Attention(Q_i W_i^Q, K_g W_g^K, V_g W_g^V)  for i in group g

其中查询头数量为h，KV头数量为h/G (G是组数)

参数对比（embed_dim=d, num_heads=h）：
- MHA: Q,K,V各需要 h 个头 → KV缓存: 2 × h × d_k
- GQA: Q需要 h 个头，K,V各需要 h/G 个头 → KV缓存: 2 × (h/G) × d_k
- MQA: Q需要 h 个头，K,V各需要 1 个头 → KV缓存: 2 × 1 × d_k

应用：Llama 2、Mistral等现代LLM
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


class GroupedQueryAttention:
    """
    Grouped-Query Attention (分组查询注意力)

    特点：
    - 查询头分成G组，每组共享一对键值头
    - 比MHA减少KV缓存（G倍），提升推理效率
    - 比MQA质量更好，保持更多表达能力
    - 是MHA和MQA的折中方案

    参数量对比（以embed_dim=512, num_heads=8为例）：
    - MHA: Q,K,V投影都是512×512，KV有8个头
    - GQA(G=2): Q投影512×512，K,V投影512×256，KV有4个头
    - GQA(G=4): Q投影512×512，K,V投影512×128，KV有2个头
    - MQA(G=8): Q投影512×512，K,V投影512×64，KV有1个头
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads=None, use_bias=True):
        """
        初始化分组查询注意力层

        Args:
            embed_dim: 嵌入维度（必须能被num_heads整除）
            num_heads: 查询头的数量（Q头数）
            num_kv_heads: 键值头的数量（K,V头数）。如果为None，则等于num_heads（退化为MHA）
            use_bias: 是否使用偏置
        """
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        # 如果未指定num_kv_heads，默认使用MHA（所有头独立）
        if num_kv_heads is None:
            num_kv_heads = num_heads

        assert num_heads % num_kv_heads == 0, "num_heads必须能被num_kv_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads  # 查询头数
        self.num_kv_heads = num_kv_heads  # 键值头数
        self.num_groups = num_heads // num_kv_heads  # 每个KV头对应几个Q头
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.use_bias = use_bias

        # Q投影矩阵：完整的num_heads个头
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # K,V投影矩阵：只有num_kv_heads个头（减少参数量）
        kv_dim = num_kv_heads * self.head_dim
        self.W_k = np.random.randn(embed_dim, kv_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, kv_dim) / np.sqrt(embed_dim)

        # 输出投影矩阵
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        if use_bias:
            self.b_q = np.zeros(embed_dim)
            self.b_k = np.zeros(kv_dim)
            self.b_v = np.zeros(kv_dim)
            self.b_o = np.zeros(embed_dim)

    def split_heads(self, x, num_heads):
        """
        将输入分割成多个头

        Args:
            x: 输入，形状为 (seq_len, num_heads * head_dim)
            num_heads: 头的数量

        Returns:
            分割后的张量，形状为 (num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[0]
        # 重塑: (seq_len, num_heads * head_dim) -> (seq_len, num_heads, head_dim)
        x = x.reshape(seq_len, num_heads, self.head_dim)
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
        K = np.dot(key, self.W_k)    # (seq_len_k, num_kv_heads * head_dim)
        V = np.dot(value, self.W_v)  # (seq_len_v, num_kv_heads * head_dim)

        if self.use_bias:
            Q += self.b_q
            K += self.b_k
            V += self.b_v

        # 步骤2: 分割成多个头
        Q = self.split_heads(Q, self.num_heads)     # (num_heads, seq_len_q, head_dim)
        K = self.split_heads(K, self.num_kv_heads)  # (num_kv_heads, seq_len_k, head_dim)
        V = self.split_heads(V, self.num_kv_heads)  # (num_kv_heads, seq_len_v, head_dim)

        # 步骤3: GQA的核心 - 每组查询头共享一对键值头
        # 将查询头按组分配，每组使用对应的KV头
        head_outputs = []
        attention_weights_list = []

        for i in range(self.num_heads):
            # 确定当前Q头属于哪个KV组
            kv_head_idx = i // self.num_groups  # 整数除法得到组索引

            # 取出第i个查询头和对应的KV头
            Q_i = Q[i]              # (seq_len_q, head_dim)
            K_i = K[kv_head_idx]    # (seq_len_k, head_dim) - 多个Q头共享同一个K头
            V_i = V[kv_head_idx]    # (seq_len_v, head_dim) - 多个Q头共享同一个V头

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
        kv_dim = self.num_kv_heads * self.head_dim

        params = {
            'W_q': self.embed_dim * self.embed_dim,
            'W_k': self.embed_dim * kv_dim,
            'W_v': self.embed_dim * kv_dim,
            'W_o': self.embed_dim * self.embed_dim,
        }

        if self.use_bias:
            params['b_q'] = self.embed_dim
            params['b_k'] = kv_dim
            params['b_v'] = kv_dim
            params['b_o'] = self.embed_dim

        total = sum(params.values())
        return total, params


class GroupedQuerySelfAttention(GroupedQueryAttention):
    """
    Grouped-Query Self-Attention (分组查询自注意力)

    是GroupedQueryAttention的特例，Q、K、V都来自同一个输入。
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
    比较不同注意力变体的参数量和KV缓存

    Args:
        embed_dim: 嵌入维度
        num_heads: 头数
        seq_len: 序列长度
    """
    head_dim = embed_dim // num_heads

    print("=" * 80)
    print("注意力变体对比分析")
    print("=" * 80)
    print(f"配置: embed_dim={embed_dim}, num_heads={num_heads}, seq_len={seq_len}")
    print(f"每个头维度: {head_dim}\n")

    variants = []

    # Multi-Head Attention (MHA)
    mha = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=num_heads)
    mha_params, _ = mha.get_num_parameters()
    mha_kv_cache = 2 * seq_len * num_heads * head_dim  # K和V的缓存
    variants.append(("MHA (标准多头)", num_heads, mha_params, mha_kv_cache, 1.0))

    # GQA with 4 KV heads
    if num_heads >= 4:
        gqa4 = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=num_heads//2)
        gqa4_params, _ = gqa4.get_num_parameters()
        gqa4_kv_cache = 2 * seq_len * (num_heads//2) * head_dim
        variants.append((f"GQA (G=2, {num_heads//2}个KV头)", num_heads//2, gqa4_params, gqa4_kv_cache,
                        gqa4_kv_cache/mha_kv_cache))

    # GQA with 2 KV heads
    if num_heads >= 4:
        gqa2 = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=num_heads//4)
        gqa2_params, _ = gqa2.get_num_parameters()
        gqa2_kv_cache = 2 * seq_len * (num_heads//4) * head_dim
        variants.append((f"GQA (G=4, {num_heads//4}个KV头)", num_heads//4, gqa2_params, gqa2_kv_cache,
                        gqa2_kv_cache/mha_kv_cache))

    # Multi-Query Attention (MQA)
    mqa = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=1)
    mqa_params, _ = mqa.get_num_parameters()
    mqa_kv_cache = 2 * seq_len * 1 * head_dim
    variants.append((f"MQA (G={num_heads}, 1个KV头)", 1, mqa_params, mqa_kv_cache, mqa_kv_cache/mha_kv_cache))

    # 打印对比表
    print(f"{'变体':<25} {'KV头数':>8} {'参数量':>15} {'KV缓存':>15} {'缓存比例':>10}")
    print("-" * 80)
    for name, kv_heads, params, kv_cache, ratio in variants:
        print(f"{name:<25} {kv_heads:>8} {params:>15,} {kv_cache:>15,} {ratio:>9.1%}")

    print("\n关键观察:")
    print(f"  ✓ MHA参数量: {mha_params:,}")
    print(f"  ✓ GQA参数量减少: {(mha_params - gqa4_params):,} (-{(1-gqa4_params/mha_params)*100:.1f}%)" if num_heads >= 4 else "")
    print(f"  ✓ MQA参数量减少: {(mha_params - mqa_params):,} (-{(1-mqa_params/mha_params)*100:.1f}%)")
    print(f"  ✓ GQA在质量和效率间取得平衡")
    print(f"  ✓ KV缓存是推理时的主要瓶颈（尤其是长序列）")


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("Grouped-Query Attention (分组查询注意力) 演示")
    print("=" * 80)

    # 参数设置
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    num_kv_heads = 2  # 使用2个KV头，8个Q头分成4组

    print(f"\n配置:")
    print(f"  序列长度: {seq_len}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  查询头数(Q): {num_heads}")
    print(f"  键值头数(K,V): {num_kv_heads}")
    print(f"  分组数: {num_heads // num_kv_heads} (每个KV头对应{num_heads // num_kv_heads}个Q头)")
    print(f"  每个头的维度: {embed_dim // num_heads}")

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 创建GQA层
    gqa = GroupedQuerySelfAttention(embed_dim, num_heads, num_kv_heads)

    # 前向传播
    print("\n" + "=" * 80)
    print("1. 标准GQA（无mask）")
    print("=" * 80)
    output, attn_weights = gqa.forward(x, return_attention=True)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"  - {num_heads}个查询头")
    print(f"  - 但只有{num_kv_heads}对不同的KV头")

    # 显示哪些Q头共享KV头
    print(f"\nKV头共享模式:")
    for kv_idx in range(num_kv_heads):
        q_heads = list(range(kv_idx * (num_heads // num_kv_heads),
                            (kv_idx + 1) * (num_heads // num_kv_heads)))
        print(f"  KV头{kv_idx} 被查询头 {q_heads} 共享")

    # 参数量分析
    print("\n" + "=" * 80)
    print("2. 参数量分析")
    print("=" * 80)

    total_params, param_breakdown = gqa.get_num_parameters()

    print(f"\nGQA参数详情:")
    for name, count in param_breakdown.items():
        print(f"  {name}: {count:,}")
    print(f"  总计: {total_params:,}")

    # 与MHA和MQA对比
    print("\n" + "=" * 80)
    print("3. 对比MHA、GQA、MQA")
    print("=" * 80)
    compare_attention_variants(embed_dim, num_heads, seq_len)

    # 因果mask示例
    print("\n" + "=" * 80)
    print("4. 带因果mask的GQA（GPT风格）")
    print("=" * 80)

    causal_mask = create_causal_mask(seq_len)
    output_masked, attn_weights_masked = gqa.forward(x, mask=causal_mask, return_attention=True)

    print(f"因果mask形状: {causal_mask.shape}")
    print(f"带mask的输出形状: {output_masked.shape}")

    # 显示分组效果
    print("\n第1组的Q头注意力权重（共享同一KV头）:")
    group_size = num_heads // num_kv_heads
    for i in range(group_size):
        print(f"  Q头{i}的注意力分布（前5个位置）: {attn_weights_masked[i, 0, :5]}")

    # 实际应用示例
    print("\n" + "=" * 80)
    print("5. 实际应用：Llama 2配置")
    print("=" * 80)

    # Llama 2 7B使用GQA
    llama_embed_dim = 4096
    llama_num_heads = 32
    llama_num_kv_heads = 8  # GQA: 32个Q头，8个KV头
    llama_seq_len = 4096

    llama_gqa = GroupedQuerySelfAttention(llama_embed_dim, llama_num_heads, llama_num_kv_heads)
    llama_params, _ = llama_gqa.get_num_parameters()

    print(f"\nLlama 2 7B 配置:")
    print(f"  嵌入维度: {llama_embed_dim}")
    print(f"  查询头数: {llama_num_heads}")
    print(f"  键值头数: {llama_num_kv_heads}")
    print(f"  分组数: {llama_num_heads // llama_num_kv_heads}")
    print(f"  序列长度: {llama_seq_len}")
    print(f"  单层注意力参数量: {llama_params:,}")

    # KV缓存对比
    head_dim = llama_embed_dim // llama_num_heads
    mha_kv_cache = 2 * llama_seq_len * llama_num_heads * head_dim
    gqa_kv_cache = 2 * llama_seq_len * llama_num_kv_heads * head_dim

    print(f"\nKV缓存对比（序列长度={llama_seq_len}）:")
    print(f"  MHA缓存: {mha_kv_cache:,} 个浮点数 ({mha_kv_cache * 4 / 1024 / 1024:.1f} MB)")
    print(f"  GQA缓存: {gqa_kv_cache:,} 个浮点数 ({gqa_kv_cache * 4 / 1024 / 1024:.1f} MB)")
    print(f"  节省: {(mha_kv_cache - gqa_kv_cache) * 4 / 1024 / 1024:.1f} MB ({(1 - gqa_kv_cache/mha_kv_cache)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("GQA的关键优势:")
    print("=" * 80)
    print("✓ 比MHA减少KV缓存，显著提升推理速度")
    print("✓ 比MQA保持更好的质量（更多KV头 = 更多表达能力）")
    print("✓ 是质量和效率的最佳平衡点")
    print("✓ 被Llama 2、Mistral等先进模型采用")
    print("✓ 特别适合长序列生成场景")
