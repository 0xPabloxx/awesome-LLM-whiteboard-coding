"""
KV Cache (键值缓存) 实现

KV Cache是自回归生成中的关键优化技术，通过缓存已计算的Key和Value，
避免重复计算，大幅提升推理速度。

核心思想：
1. 在自回归生成时，每个新token都要关注之前的所有token
2. 之前token的K和V在每步都相同，无需重复计算
3. 只需计算新token的K和V，并拼接到缓存中
4. 显著减少计算量（从O(n²)到O(n)）

优势：
- 推理速度提升：约2-3倍（长序列更明显）
- 计算量减少：避免重复计算之前token的K、V
- 显存换速度：用额外显存存储缓存，换取计算加速

应用：所有自回归生成模型（GPT、LLaMA、ChatGPT等）
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class KVCache:
    """
    KV缓存实现

    用于存储和管理多头注意力的Key和Value缓存。
    """

    def __init__(self, num_heads, head_dim, max_seq_len=2048):
        """
        初始化KV缓存

        Args:
            num_heads: 注意力头数
            head_dim: 每个头的维度
            max_seq_len: 最大序列长度（预分配缓存空间）
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # 预分配缓存空间 (num_heads, max_seq_len, head_dim)
        self.k_cache = np.zeros((num_heads, max_seq_len, head_dim))
        self.v_cache = np.zeros((num_heads, max_seq_len, head_dim))

        # 当前缓存的实际长度
        self.cache_len = 0

    def update(self, new_k, new_v):
        """
        更新缓存（添加新的K和V）

        Args:
            new_k: 新的Key，形状为 (num_heads, new_seq_len, head_dim)
            new_v: 新的Value，形状为 (num_heads, new_seq_len, head_dim)

        Returns:
            full_k: 完整的Key（包括缓存），形状为 (num_heads, total_len, head_dim)
            full_v: 完整的Value（包括缓存），形状为 (num_heads, total_len, head_dim)
        """
        new_seq_len = new_k.shape[1]

        # 检查是否超出最大长度
        if self.cache_len + new_seq_len > self.max_seq_len:
            raise ValueError(f"序列长度超出最大值 {self.max_seq_len}")

        # 将新的K、V添加到缓存
        self.k_cache[:, self.cache_len:self.cache_len + new_seq_len, :] = new_k
        self.v_cache[:, self.cache_len:self.cache_len + new_seq_len, :] = new_v

        # 更新缓存长度
        self.cache_len += new_seq_len

        # 返回完整的K、V（从缓存中提取）
        full_k = self.k_cache[:, :self.cache_len, :]
        full_v = self.v_cache[:, :self.cache_len, :]

        return full_k, full_v

    def get_cache(self):
        """
        获取当前缓存

        Returns:
            k_cache: 当前的Key缓存
            v_cache: 当前的Value缓存
        """
        return self.k_cache[:, :self.cache_len, :], self.v_cache[:, :self.cache_len, :]

    def reset(self):
        """重置缓存"""
        self.cache_len = 0
        self.k_cache.fill(0)
        self.v_cache.fill(0)


class MultiHeadAttentionWithKVCache:
    """
    带KV Cache的多头注意力

    支持两种模式：
    1. 预填充（Prefill）：处理输入prompt，计算所有token的KV
    2. 解码（Decode）：生成新token，使用KV缓存
    """

    def __init__(self, embed_dim, num_heads, max_seq_len=2048):
        """
        初始化

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
        """
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q、K、V投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # 输出投影
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

        # KV缓存
        self.kv_cache = KVCache(num_heads, self.head_dim, max_seq_len)

    def split_heads(self, x):
        """分割成多个头"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        return x

    def combine_heads(self, x):
        """合并多个头"""
        x = x.transpose(1, 0, 2)  # (seq_len, num_heads, head_dim)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.embed_dim)
        return x

    def forward_without_cache(self, x, mask=None):
        """
        无缓存的标准前向传播（用于对比）

        Args:
            x: 输入，形状为 (seq_len, embed_dim)
            mask: 注意力mask

        Returns:
            output: 输出，形状为 (seq_len, embed_dim)
        """
        seq_len = x.shape[0]

        # 投影到Q、K、V
        Q = np.dot(x, self.W_q.T)
        K = np.dot(x, self.W_k.T)
        V = np.dot(x, self.W_v.T)

        # 分割成多个头
        Q = self.split_heads(Q)  # (num_heads, seq_len, head_dim)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算注意力（对每个头）
        outputs = []
        for i in range(self.num_heads):
            Q_i = Q[i]  # (seq_len, head_dim)
            K_i = K[i]
            V_i = V[i]

            # Scaled dot-product attention
            scores = np.dot(Q_i, K_i.T) / np.sqrt(self.head_dim)

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

    def forward_with_cache(self, x, use_cache=True, is_prefill=False):
        """
        带KV缓存的前向传播

        Args:
            x: 输入，形状为 (new_seq_len, embed_dim)
               - Prefill阶段: new_seq_len = prompt_len
               - Decode阶段: new_seq_len = 1 (单个新token)
            use_cache: 是否使用缓存
            is_prefill: 是否是预填充阶段（处理prompt）

        Returns:
            output: 输出，形状为 (new_seq_len, embed_dim)
        """
        new_seq_len = x.shape[0]

        # 计算新token的Q、K、V
        Q = np.dot(x, self.W_q.T)
        K_new = np.dot(x, self.W_k.T)
        V_new = np.dot(x, self.W_v.T)

        # 分割成多个头
        Q = self.split_heads(Q)  # (num_heads, new_seq_len, head_dim)
        K_new = self.split_heads(K_new)
        V_new = self.split_heads(V_new)

        if use_cache:
            # 更新缓存，获取完整的K、V
            K_full, V_full = self.kv_cache.update(K_new, V_new)
        else:
            K_full, V_full = K_new, V_new

        # 计算注意力
        outputs = []
        for i in range(self.num_heads):
            Q_i = Q[i]  # (new_seq_len, head_dim)
            K_i = K_full[i]  # (total_len, head_dim)
            V_i = V_full[i]  # (total_len, head_dim)

            # Attention scores
            scores = np.dot(Q_i, K_i.T) / np.sqrt(self.head_dim)

            # 对于decode阶段，自动应用causal mask
            if use_cache and not is_prefill:
                # 新token只能看到自己和之前的token
                total_len = K_i.shape[0]
                mask = np.tril(np.ones((new_seq_len, total_len)))
                scores = np.where(mask == 0, -1e9, scores)

            attn_weights = softmax(scores, axis=-1)
            output_i = np.dot(attn_weights, V_i)

            outputs.append(output_i)

        # 合并头
        multi_head_output = np.stack(outputs, axis=0)
        concatenated = self.combine_heads(multi_head_output)

        # 输出投影
        output = np.dot(concatenated, self.W_o.T)

        return output


def autoregressive_generation_with_cache(model, prompt_tokens, max_new_tokens=10, temperature=1.0):
    """
    使用KV Cache的自回归生成

    Args:
        model: MultiHeadAttentionWithKVCache模型
        prompt_tokens: prompt的token表示，形状为 (prompt_len, embed_dim)
        max_new_tokens: 生成的新token数量
        temperature: 采样温度

    Returns:
        generated_sequence: 生成的完整序列（包括prompt）
    """
    # 重置缓存
    model.kv_cache.reset()

    # 阶段1: Prefill - 处理prompt
    print(f"Prefill阶段: 处理 {prompt_tokens.shape[0]} 个prompt tokens")
    _ = model.forward_with_cache(prompt_tokens, use_cache=True, is_prefill=True)

    # 阶段2: Decode - 逐个生成新token
    print(f"Decode阶段: 生成 {max_new_tokens} 个新tokens")

    generated_tokens = []
    current_token = prompt_tokens[-1:, :]  # 从最后一个prompt token开始

    for step in range(max_new_tokens):
        # 生成下一个token
        output = model.forward_with_cache(current_token, use_cache=True, is_prefill=False)

        # 这里简化处理：直接使用输出作为下一个token的embedding
        # 实际应用中需要通过语言模型头得到logits，然后采样
        next_token = output  # (1, embed_dim)

        generated_tokens.append(next_token)

        # 更新current_token为新生成的token
        current_token = next_token

        print(f"  Step {step + 1}/{max_new_tokens}: 缓存长度 = {model.kv_cache.cache_len}")

    # 拼接生成的序列
    generated_sequence = np.concatenate([prompt_tokens] + generated_tokens, axis=0)

    return generated_sequence


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("KV Cache (键值缓存) 演示")
    print("=" * 70)

    # ========== 1. KVCache基础演示 ==========
    print("\n" + "=" * 70)
    print("1. KVCache基础操作")
    print("=" * 70)

    num_heads = 8
    head_dim = 64
    max_seq_len = 100

    kv_cache = KVCache(num_heads, head_dim, max_seq_len)

    print(f"\n配置:")
    print(f"  注意力头数: {num_heads}")
    print(f"  每个头维度: {head_dim}")
    print(f"  最大序列长度: {max_seq_len}")
    print(f"  初始缓存长度: {kv_cache.cache_len}")

    # 第一次更新：添加prompt的K、V
    prompt_len = 10
    k1 = np.random.randn(num_heads, prompt_len, head_dim)
    v1 = np.random.randn(num_heads, prompt_len, head_dim)

    full_k, full_v = kv_cache.update(k1, v1)

    print(f"\n第一次更新（Prefill）:")
    print(f"  添加 {prompt_len} 个token的K、V")
    print(f"  当前缓存长度: {kv_cache.cache_len}")
    print(f"  完整K形状: {full_k.shape}")
    print(f"  完整V形状: {full_v.shape}")

    # 第二次更新：添加新生成token的K、V
    k2 = np.random.randn(num_heads, 1, head_dim)
    v2 = np.random.randn(num_heads, 1, head_dim)

    full_k, full_v = kv_cache.update(k2, v2)

    print(f"\n第二次更新（Decode step 1）:")
    print(f"  添加 1 个新token的K、V")
    print(f"  当前缓存长度: {kv_cache.cache_len}")
    print(f"  完整K形状: {full_k.shape}")

    # ========== 2. 计算量对比 ==========
    print("\n" + "=" * 70)
    print("2. 无缓存 vs 有缓存的计算量对比")
    print("=" * 70)

    seq_lengths = [1, 10, 50, 100, 500, 1000]

    print(f"\n{'序列长度':<12} {'无缓存计算':<20} {'有缓存计算':<20} {'加速比':<10}")
    print("-" * 70)

    for seq_len in seq_lengths:
        # 无缓存：每次都计算所有token的Q、K、V
        # 计算量 ∝ seq_len² （attention矩阵）
        no_cache_ops = seq_len * seq_len

        # 有缓存：只计算新token的Q，K、V来自缓存
        # Prefill: seq_len² （一次性）
        # Decode: seq_len × 1 × seq_len_per_step （每步）
        # 总计（生成n个token）: seq_len² + n×seq_len
        # 平均每token: seq_len + 1
        with_cache_ops = seq_len + 1

        speedup = no_cache_ops / with_cache_ops

        print(f"{seq_len:<12} {no_cache_ops:<20,} {with_cache_ops:<20,} {speedup:<10.1f}x")

    # ========== 3. 带KV Cache的多头注意力 ==========
    print("\n" + "=" * 70)
    print("3. 带KV Cache的多头注意力")
    print("=" * 70)

    embed_dim = 512
    num_heads = 8
    prompt_len = 20
    new_tokens = 5

    # 创建模型
    mha = MultiHeadAttentionWithKVCache(embed_dim, num_heads, max_seq_len=1024)

    print(f"\n配置:")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  生成新token数: {new_tokens}")

    # 生成prompt
    prompt = np.random.randn(prompt_len, embed_dim)

    # 方法1: 不使用缓存（标准方法）
    print(f"\n方法1: 不使用KV Cache")
    import time

    mha.kv_cache.reset()

    start = time.time()
    output_no_cache = mha.forward_without_cache(prompt)
    time_no_cache = time.time() - start

    print(f"  输出形状: {output_no_cache.shape}")
    print(f"  计算时间: {time_no_cache*1000:.3f}ms")

    # 方法2: 使用缓存
    print(f"\n方法2: 使用KV Cache")

    mha.kv_cache.reset()

    start = time.time()
    # Prefill阶段
    output_prefill = mha.forward_with_cache(prompt, use_cache=True, is_prefill=True)

    # Decode阶段（模拟生成new_tokens个token）
    current_token = prompt[-1:, :]
    for _ in range(new_tokens):
        output_decode = mha.forward_with_cache(current_token, use_cache=True, is_prefill=False)
        current_token = output_decode

    time_with_cache = time.time() - start

    print(f"  Prefill输出形状: {output_prefill.shape}")
    print(f"  Decode输出形状: {output_decode.shape}")
    print(f"  总计算时间: {time_with_cache*1000:.3f}ms")
    print(f"  最终缓存长度: {mha.kv_cache.cache_len}")

    # ========== 4. 自回归生成演示 ==========
    print("\n" + "=" * 70)
    print("4. 自回归生成（完整流程）")
    print("=" * 70)

    prompt_len = 15
    max_new_tokens = 10

    prompt_tokens = np.random.randn(prompt_len, embed_dim)

    mha_gen = MultiHeadAttentionWithKVCache(embed_dim, num_heads, max_seq_len=1024)

    generated = autoregressive_generation_with_cache(
        mha_gen,
        prompt_tokens,
        max_new_tokens=max_new_tokens
    )

    print(f"\n生成结果:")
    print(f"  Prompt长度: {prompt_len}")
    print(f"  生成新token数: {max_new_tokens}")
    print(f"  完整序列长度: {generated.shape[0]}")
    print(f"  最终缓存长度: {mha_gen.kv_cache.cache_len}")

    # ========== 5. 显存占用分析 ==========
    print("\n" + "=" * 70)
    print("5. KV Cache显存占用分析")
    print("=" * 70)

    batch_size = 1
    seq_len = 2048  # 最大序列长度
    num_layers = 32  # GPT-3 style

    # 每层的KV缓存大小（字节）
    # K: (num_heads, seq_len, head_dim)
    # V: (num_heads, seq_len, head_dim)
    # 总计: 2 × num_heads × seq_len × head_dim × sizeof(float32)
    kv_cache_per_layer = 2 * num_heads * seq_len * head_dim * 4  # float32 = 4 bytes

    total_kv_cache = num_layers * kv_cache_per_layer / (1024 ** 3)  # GB

    print(f"\n配置（GPT-3风格）:")
    print(f"  层数: {num_layers}")
    print(f"  注意力头数: {num_heads}")
    print(f"  每头维度: {head_dim}")
    print(f"  最大序列长度: {seq_len}")
    print(f"  Batch大小: {batch_size}")

    print(f"\nKV Cache显存占用:")
    print(f"  每层: {kv_cache_per_layer / (1024**2):.2f} MB")
    print(f"  总计({num_layers}层): {total_kv_cache:.2f} GB")

    print(f"\n对于batch_size={batch_size}:")
    print(f"  总KV Cache: {total_kv_cache * batch_size:.2f} GB")

    # 不同batch size的对比
    print(f"\n不同batch size的显存占用:")
    for bs in [1, 2, 4, 8, 16, 32]:
        mem = total_kv_cache * bs
        print(f"  Batch {bs}: {mem:.2f} GB")

    # ========== 6. 优化技巧 ==========
    print("\n" + "=" * 70)
    print("6. KV Cache优化技巧")
    print("=" * 70)

    print("\n优化方法:")
    print("  1. Multi-Query Attention (MQA)")
    print("     - 所有头共享同一个K、V")
    print("     - 显存减少: num_heads倍")

    print("\n  2. Grouped-Query Attention (GQA)")
    print("     - 多个头共享一组K、V")
    print("     - 平衡效果和显存")
    print("     - LLaMA-2采用")

    print("\n  3. 量化KV Cache")
    print("     - 使用int8/int4存储K、V")
    print("     - 显存减少: 2-4倍")
    print("     - 精度损失较小")

    print("\n  4. Paged Attention (vLLM)")
    print("     - 按块管理KV缓存")
    print("     - 提高显存利用率")
    print("     - 支持动态batch")

    # ========== 7. 总结 ==========
    print("\n" + "=" * 70)
    print("KV Cache的关键特性:")
    print("=" * 70)
    print("✓ 自回归生成的标准优化技术")
    print("✓ 避免重复计算，推理速度提升2-3倍")
    print("✓ 显存换速度：需要额外显存存储K、V")
    print("✓ 对长序列生成效果尤其明显")
    print("✓ 所有现代LLM（GPT、LLaMA等）都使用")
    print("✓ 需要配合其他技术（MQA、GQA、量化）优化显存")
