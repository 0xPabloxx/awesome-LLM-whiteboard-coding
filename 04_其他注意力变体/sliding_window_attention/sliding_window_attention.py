"""
Sliding Window Attention (滑动窗口注意力) 实现

滑动窗口注意力是一种高效的注意力机制，每个token只关注固定窗口内的邻近token，
而不是全序列的所有token。由Beltagy等人在Longformer论文中推广，
并在Mistral、Longformer等模型中被广泛采用。

核心思想：
1. 限制注意力范围到固定大小的窗口
2. 显著降低计算复杂度（从O(n²)到O(n×w)）
3. 保留局部上下文信息
4. 适合处理长序列

公式：
对于位置i，只计算窗口内位置的注意力：
  Attention_i = softmax(Q_i · K_{[i-w/2:i+w/2]}) · V_{[i-w/2:i+w/2]}

其中w是窗口大小

优势：
- 计算效率高：线性复杂度O(n×w)
- 内存占用少：只存储窗口内的注意力
- 局部建模能力强：捕获邻近依赖
- 可扩展到超长序列

应用：Longformer、BigBird、Mistral等长序列模型
"""

import numpy as np


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_sliding_window_mask(seq_len, window_size):
    """
    创建滑动窗口mask

    Args:
        seq_len: 序列长度
        window_size: 窗口大小（每侧的范围）

    Returns:
        mask: 滑动窗口mask，形状为 (seq_len, seq_len)

    说明：
        对于位置i，可以看到[i-window_size, i+window_size]范围内的位置
        窗口总大小为2*window_size+1（包括自己）
    """
    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        # 计算窗口范围
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1

    return mask


def create_sliding_window_mask_one_sided(seq_len, window_size):
    """
    创建单侧滑动窗口mask（只看左侧）

    Args:
        seq_len: 序列长度
        window_size: 窗口大小

    Returns:
        mask: 单侧滑动窗口mask

    说明：
        对于位置i，只能看到[i-window_size, i]范围内的位置
        用于自回归生成
    """
    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = i + 1  # 包括自己
        mask[i, start:end] = 1

    return mask


class SlidingWindowAttention:
    """
    Sliding Window Attention (滑动窗口注意力)

    特点：
    - 每个token只关注固定窗口内的token
    - 计算复杂度从O(n²)降到O(n×w)
    - 内存占用显著减少
    - 适合处理长序列
    """

    def __init__(self, embed_dim, num_heads, window_size, use_bias=True):
        """
        初始化滑动窗口注意力层

        Args:
            embed_dim: 嵌入维度（必须能被num_heads整除）
            num_heads: 注意力头的数量
            window_size: 窗口大小（每侧的范围）
            use_bias: 是否使用偏置
        """
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
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

    def split_heads(self, x):
        """
        将输入分割成多个头

        Args:
            x: 输入，形状为 (seq_len, embed_dim)

        Returns:
            分割后的张量，形状为 (num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
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
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.embed_dim)
        return x

    def sliding_window_attention(self, Q, K, V, window_mask=None):
        """
        计算滑动窗口注意力

        Args:
            Q: Query，形状为 (num_heads, seq_len, head_dim)
            K: Key，形状为 (num_heads, seq_len, head_dim)
            V: Value，形状为 (num_heads, seq_len, head_dim)
            window_mask: 窗口mask，形状为 (seq_len, seq_len)

        Returns:
            output: 输出，形状为 (num_heads, seq_len, head_dim)
            attention_weights: 注意力权重（稀疏）
        """
        num_heads, seq_len, head_dim = Q.shape

        # 1. 计算注意力得分: Q @ K^T
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(head_dim)
        # scores: (num_heads, seq_len, seq_len)

        # 2. 应用滑动窗口mask
        if window_mask is None:
            window_mask = create_sliding_window_mask(seq_len, self.window_size)

        # 扩展mask到所有头
        window_mask = window_mask[None, :, :]  # (1, seq_len, seq_len)

        # 将窗口外的位置设为-inf，使softmax后为0
        scores = np.where(window_mask == 0, -1e9, scores)

        # 3. Softmax归一化
        attention_weights = softmax(scores, axis=-1)

        # 4. 加权求和V
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key=None, value=None, window_mask=None, return_attention=False):
        """
        前向传播

        Args:
            query: Query输入，形状为 (seq_len, embed_dim)
            key: Key输入，形状为 (seq_len, embed_dim)。如果为None，则使用query（自注意力）
            value: Value输入，形状为 (seq_len, embed_dim)。如果为None，则使用key
            window_mask: 自定义窗口mask。如果为None，则使用默认双向窗口
            return_attention: 是否返回注意力权重

        Returns:
            output: 输出，形状为 (seq_len, embed_dim)
            attention_weights: (可选) 注意力权重
        """
        if key is None:
            key = query
        if value is None:
            value = key

        seq_len = query.shape[0]

        # 1. 线性投影得到Q、K、V
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)

        if self.use_bias:
            Q += self.b_q
            K += self.b_k
            V += self.b_v

        # 2. 分割成多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 滑动窗口注意力
        multi_head_output, attention_weights = self.sliding_window_attention(
            Q, K, V, window_mask=window_mask
        )

        # 4. 合并头
        concatenated = self.combine_heads(multi_head_output)

        # 5. 输出投影
        output = np.dot(concatenated, self.W_o)
        if self.use_bias:
            output += self.b_o

        if return_attention:
            return output, attention_weights

        return output


class SlidingWindowSelfAttention(SlidingWindowAttention):
    """
    Sliding Window Self-Attention (滑动窗口自注意力)

    是SlidingWindowAttention的特例，Q、K、V都来自同一个输入。
    """

    def forward(self, x, window_mask=None, return_attention=False):
        """
        前向传播（自注意力版本）

        Args:
            x: 输入序列，形状为 (seq_len, embed_dim)
            window_mask: 窗口mask
            return_attention: 是否返回注意力权重

        Returns:
            output: 输出，形状为 (seq_len, embed_dim)
            attention_weights: (可选) 注意力权重
        """
        return super().forward(x, x, x, window_mask=window_mask, return_attention=return_attention)


def visualize_window_mask(seq_len, window_size):
    """
    可视化滑动窗口mask

    Args:
        seq_len: 序列长度
        window_size: 窗口大小
    """
    mask = create_sliding_window_mask(seq_len, window_size)

    print(f"滑动窗口Mask (seq_len={seq_len}, window_size={window_size}):")
    print(mask.astype(int))
    print("\n说明:")
    print(f"  - 每个位置可以看到前后{window_size}个位置")
    print(f"  - 窗口总大小: {2*window_size+1}（包括自己）")
    print(f"  - 1表示可见，0表示被mask")

    # 统计每个位置的可见范围
    visible_counts = mask.sum(axis=1)
    print(f"\n每个位置的可见token数:")
    print(f"  最小: {int(visible_counts.min())}")
    print(f"  最大: {int(visible_counts.max())}")
    print(f"  平均: {visible_counts.mean():.1f}")


def compare_complexity(seq_lengths, window_size):
    """
    比较滑动窗口注意力和标准注意力的复杂度

    Args:
        seq_lengths: 序列长度列表
        window_size: 窗口大小
    """
    print("=" * 80)
    print("计算复杂度对比：滑动窗口 vs 标准注意力")
    print("=" * 80)
    print(f"\n窗口大小: {window_size}\n")
    print(f"{'序列长度':>10} {'标准注意力':>15} {'滑动窗口':>15} {'加速比':>10} {'内存节省':>10}")
    print("-" * 65)

    for n in seq_lengths:
        standard_ops = n * n  # O(n²)
        sliding_ops = n * (2 * window_size + 1)  # O(n × w)
        speedup = standard_ops / sliding_ops
        memory_saving = (1 - sliding_ops / standard_ops) * 100

        print(f"{n:>10} {standard_ops:>15,} {sliding_ops:>15,} "
              f"{speedup:>9.1f}x {memory_saving:>9.1f}%")

    print("\n关键观察:")
    print(f"  ✓ 序列越长，滑动窗口的优势越明显")
    print(f"  ✓ 窗口大小固定，复杂度与序列长度线性相关")
    print(f"  ✓ 内存占用大幅降低，可处理更长序列")


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("Sliding Window Attention (滑动窗口注意力) 演示")
    print("=" * 80)

    # 参数设置
    seq_len = 16
    embed_dim = 64
    num_heads = 4
    window_size = 2  # 每侧看2个位置，总窗口大小=5

    print(f"\n配置:")
    print(f"  序列长度: {seq_len}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  窗口大小: {window_size} (每侧)")
    print(f"  总窗口大小: {2*window_size+1} (包括自己)")

    # 1. 可视化窗口mask
    print("\n" + "=" * 80)
    print("1. 滑动窗口Mask可视化")
    print("=" * 80)
    visualize_window_mask(seq_len, window_size)

    # 2. 创建滑动窗口注意力层
    print("\n" + "=" * 80)
    print("2. 滑动窗口注意力计算")
    print("=" * 80)

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 创建滑动窗口注意力层
    swa = SlidingWindowSelfAttention(embed_dim, num_heads, window_size)

    # 前向传播
    output, attn_weights = swa.forward(x, return_attention=True)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")

    # 3. 分析注意力模式
    print("\n" + "=" * 80)
    print("3. 注意力模式分析（第1个头）")
    print("=" * 80)

    print("\n注意力权重矩阵（稀疏）:")
    print(attn_weights[0])
    print("\n说明:")
    print("  - 非零元素集中在对角线附近")
    print("  - 窗口外的位置注意力权重为0")
    print("  - 稀疏性带来计算和内存优势")

    # 统计稀疏度
    total_elements = attn_weights[0].size
    nonzero_elements = np.count_nonzero(attn_weights[0] > 1e-6)
    sparsity = (1 - nonzero_elements / total_elements) * 100

    print(f"\n稀疏度统计:")
    print(f"  总元素数: {total_elements}")
    print(f"  非零元素数: {nonzero_elements}")
    print(f"  稀疏度: {sparsity:.1f}%")

    # 4. 单侧窗口（自回归）
    print("\n" + "=" * 80)
    print("4. 单侧滑动窗口（自回归生成）")
    print("=" * 80)

    # 创建单侧窗口mask
    one_sided_mask = create_sliding_window_mask_one_sided(seq_len, window_size)

    print(f"单侧窗口mask (只看左侧{window_size}个位置):")
    print(one_sided_mask.astype(int))

    # 使用单侧窗口
    output_one_sided, attn_weights_one_sided = swa.forward(
        x, window_mask=one_sided_mask, return_attention=True
    )

    print(f"\n单侧窗口注意力权重（第1个头）:")
    print(attn_weights_one_sided[0])
    print("\n用途:")
    print("  • 自回归语言模型（GPT风格）")
    print("  • 限制只能看到历史信息")
    print("  • 结合滑动窗口和因果mask")

    # 5. 复杂度对比
    print("\n" + "=" * 80)
    print("5. 计算复杂度分析")
    print("=" * 80)

    test_lengths = [128, 512, 1024, 4096, 16384]
    compare_complexity(test_lengths, window_size=128)

    # 6. 不同窗口大小的对比
    print("\n" + "=" * 80)
    print("6. 不同窗口大小的影响")
    print("=" * 80)

    test_seq_len = 20
    window_sizes = [1, 2, 4, 8]

    print(f"序列长度: {test_seq_len}\n")
    print(f"{'窗口大小':>10} {'窗口总大小':>12} {'可见token':>12} {'稀疏度':>10}")
    print("-" * 50)

    for ws in window_sizes:
        total_window = 2 * ws + 1
        # 计算平均可见token数（考虑边界）
        mask = create_sliding_window_mask(test_seq_len, ws)
        avg_visible = mask.sum() / test_seq_len
        sparsity_pct = (1 - avg_visible / test_seq_len) * 100

        print(f"{ws:>10} {total_window:>12} {avg_visible:>12.1f} {sparsity_pct:>9.1f}%")

    print("\n权衡:")
    print("  • 窗口越小：计算越快，但上下文信息越少")
    print("  • 窗口越大：上下文越丰富，但计算成本越高")
    print("  • Mistral 7B使用窗口大小4096")

    # 7. 实际应用：Mistral配置
    print("\n" + "=" * 80)
    print("7. 实际应用：Mistral 7B配置")
    print("=" * 80)

    # Mistral 7B使用滑动窗口注意力
    mistral_embed_dim = 4096
    mistral_num_heads = 32
    mistral_window_size = 4096  # 滑动窗口大小
    mistral_seq_len = 32768  # 支持的序列长度

    print(f"\nMistral 7B配置:")
    print(f"  嵌入维度: {mistral_embed_dim}")
    print(f"  注意力头数: {mistral_num_heads}")
    print(f"  窗口大小: {mistral_window_size}")
    print(f"  支持序列长度: {mistral_seq_len}")

    # 计算复杂度
    standard_ops = mistral_seq_len * mistral_seq_len
    sliding_ops = mistral_seq_len * (2 * mistral_window_size + 1)
    speedup = standard_ops / sliding_ops

    print(f"\n复杂度对比:")
    print(f"  标准注意力: {standard_ops:,} 操作")
    print(f"  滑动窗口: {sliding_ops:,} 操作")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  内存节省: {(1 - sliding_ops/standard_ops)*100:.1f}%")

    print(f"\nMistral的设计选择:")
    print(f"  ✓ 窗口大小{mistral_window_size}提供足够的局部上下文")
    print(f"  ✓ 可以处理{mistral_seq_len}长度的序列")
    print(f"  ✓ 比标准注意力快{speedup:.0f}倍")
    print(f"  ✓ 适合长文档处理和对话场景")

    # 8. 混合策略：全局+局部
    print("\n" + "=" * 80)
    print("8. 混合策略：全局token + 滑动窗口")
    print("=" * 80)

    print("\nLongformer混合策略:")
    print("  • 大部分token使用滑动窗口（局部注意力）")
    print("  • 少数特殊token使用全局注意力（如[CLS]）")
    print("  • 平衡局部和全局信息")

    # 示例：创建混合mask
    test_len = 10
    mixed_mask = create_sliding_window_mask(test_len, window_size=1)
    # 让第0个位置（如[CLS]）能看到所有位置
    mixed_mask[0, :] = 1
    # 让所有位置都能看到第0个位置
    mixed_mask[:, 0] = 1

    print(f"\n混合mask示例 (位置0为全局):")
    print(mixed_mask.astype(int))
    print("\n说明:")
    print("  • 第0行全为1: [CLS]可以看到所有token")
    print("  • 第0列全为1: 所有token都能看到[CLS]")
    print("  • 其他位置使用滑动窗口")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("滑动窗口注意力通过限制注意力范围，")
    print("在保持局部建模能力的同时大幅降低计算复杂度，")
    print("是处理长序列的关键技术之一。")
    print("\n关键特点:")
    print("  ✓ 线性复杂度 O(n×w)")
    print("  ✓ 内存占用小")
    print("  ✓ 可扩展到超长序列")
    print("  ✓ 保留局部上下文信息")
