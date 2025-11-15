"""
Sparse Attention (稀疏注意力) 实现

稀疏注意力只计算部分位置对的注意力，而不是所有n²个位置对。
通过精心设计的稀疏模式，在保持性能的同时大幅降低计算复杂度。

核心思想：
1. 标准注意力计算所有位置对：O(n²)
2. 稀疏注意力只计算有意义的位置对：O(n×k)，其中k是稀疏度
3. 常见模式：局部窗口、全局token、随机采样

稀疏模式类型：
1. Local (局部): 每个位置只关注窗口内的邻居
2. Global (全局): 特定位置可以关注所有位置
3. Random (随机): 随机选择一些位置对
4. Strided (步进): 按固定步长采样

应用：Longformer、BigBird、Sparse Transformer等
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_local_mask(seq_len, window_size):
    """
    创建局部注意力mask（滑动窗口）

    每个位置只关注前后window_size范围内的位置

    Args:
        seq_len: 序列长度
        window_size: 窗口大小（单侧）

    Returns:
        mask: 稀疏mask矩阵 (seq_len, seq_len)
    """
    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        # 窗口范围：[i-window_size, i+window_size]
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1

    return mask


def create_global_mask(seq_len, global_indices):
    """
    创建全局注意力mask

    特定位置可以关注和被关注所有位置
    通常用于特殊token（如[CLS]、[SEP]）

    Args:
        seq_len: 序列长度
        global_indices: 全局位置的索引列表

    Returns:
        mask: 包含全局注意力的mask (seq_len, seq_len)
    """
    mask = np.zeros((seq_len, seq_len))

    for idx in global_indices:
        # 全局位置可以关注所有位置
        mask[idx, :] = 1
        # 所有位置可以关注全局位置
        mask[:, idx] = 1

    return mask


def create_random_mask(seq_len, num_random):
    """
    创建随机注意力mask

    每个位置随机关注num_random个其他位置

    Args:
        seq_len: 序列长度
        num_random: 每个位置随机关注的数量

    Returns:
        mask: 随机稀疏mask (seq_len, seq_len)
    """
    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        # 随机选择num_random个位置
        random_indices = np.random.choice(seq_len, size=num_random, replace=False)
        mask[i, random_indices] = 1
        # 确保自己总是可见
        mask[i, i] = 1

    return mask


def create_strided_mask(seq_len, stride):
    """
    创建步进注意力mask

    每个位置关注间隔stride的位置

    Args:
        seq_len: 序列长度
        stride: 步长

    Returns:
        mask: 步进稀疏mask (seq_len, seq_len)
    """
    mask = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        # 步进采样
        indices = list(range(i % stride, seq_len, stride))
        mask[i, indices] = 1

    return mask


def create_longformer_mask(seq_len, window_size, global_indices):
    """
    创建Longformer风格的稀疏mask

    组合局部窗口 + 全局注意力

    Args:
        seq_len: 序列长度
        window_size: 局部窗口大小
        global_indices: 全局token的位置

    Returns:
        mask: Longformer稀疏mask (seq_len, seq_len)
    """
    # 局部窗口
    local_mask = create_local_mask(seq_len, window_size)

    # 全局token
    global_mask = create_global_mask(seq_len, global_indices)

    # 组合
    mask = np.maximum(local_mask, global_mask)

    return mask


def create_bigbird_mask(seq_len, window_size, num_random, global_indices):
    """
    创建BigBird风格的稀疏mask

    组合局部窗口 + 全局注意力 + 随机注意力

    Args:
        seq_len: 序列长度
        window_size: 局部窗口大小
        num_random: 每个位置的随机连接数
        global_indices: 全局token的位置

    Returns:
        mask: BigBird稀疏mask (seq_len, seq_len)
    """
    # 局部窗口
    local_mask = create_local_mask(seq_len, window_size)

    # 全局token
    global_mask = create_global_mask(seq_len, global_indices)

    # 随机连接
    random_mask = create_random_mask(seq_len, num_random)

    # 组合所有模式
    mask = np.maximum(np.maximum(local_mask, global_mask), random_mask)

    return mask


class SparseAttention:
    """
    稀疏注意力机制

    支持多种稀疏模式：
    - local: 局部窗口
    - global: 全局token
    - random: 随机连接
    - longformer: 局部+全局
    - bigbird: 局部+全局+随机
    """

    def __init__(self, embed_dim, pattern='local', **kwargs):
        """
        初始化稀疏注意力层

        Args:
            embed_dim: 嵌入维度
            pattern: 稀疏模式类型
            **kwargs: 模式相关的参数
                - window_size: 窗口大小（local, longformer, bigbird）
                - global_indices: 全局位置（global, longformer, bigbird）
                - num_random: 随机连接数（random, bigbird）
                - stride: 步长（strided）
        """
        self.embed_dim = embed_dim
        self.pattern = pattern
        self.kwargs = kwargs

        # 初始化Q、K、V投影矩阵
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def create_mask(self, seq_len):
        """根据pattern创建稀疏mask"""
        if self.pattern == 'local':
            window_size = self.kwargs.get('window_size', 3)
            return create_local_mask(seq_len, window_size)

        elif self.pattern == 'global':
            global_indices = self.kwargs.get('global_indices', [0])
            return create_global_mask(seq_len, global_indices)

        elif self.pattern == 'random':
            num_random = self.kwargs.get('num_random', 5)
            return create_random_mask(seq_len, num_random)

        elif self.pattern == 'strided':
            stride = self.kwargs.get('stride', 2)
            return create_strided_mask(seq_len, stride)

        elif self.pattern == 'longformer':
            window_size = self.kwargs.get('window_size', 3)
            global_indices = self.kwargs.get('global_indices', [0])
            return create_longformer_mask(seq_len, window_size, global_indices)

        elif self.pattern == 'bigbird':
            window_size = self.kwargs.get('window_size', 3)
            num_random = self.kwargs.get('num_random', 3)
            global_indices = self.kwargs.get('global_indices', [0])
            return create_bigbird_mask(seq_len, window_size, num_random, global_indices)

        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列，形状 (seq_len, embed_dim)

        Returns:
            output: 输出序列，形状 (seq_len, embed_dim)
            mask: 使用的稀疏mask
            sparsity: 稀疏度（0-1之间，越小越稀疏）
        """
        seq_len, embed_dim = x.shape

        # 线性投影
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 创建稀疏mask
        mask = self.create_mask(seq_len)

        # 计算注意力得分
        scores = np.dot(Q, K.T) / np.sqrt(embed_dim)

        # 应用稀疏mask（mask为0的位置设为-inf）
        scores = np.where(mask == 1, scores, -1e9)

        # Softmax
        attention_weights = softmax(scores, axis=-1)

        # 加权求和
        output = np.dot(attention_weights, V)

        # 计算稀疏度
        sparsity = np.sum(mask) / (seq_len * seq_len)

        return output, mask, sparsity


def visualize_sparse_patterns(seq_len=64):
    """
    可视化不同的稀疏模式

    Args:
        seq_len: 序列长度
    """
    patterns = {
        'Local (窗口=3)': create_local_mask(seq_len, window_size=3),
        'Global (位置0)': create_global_mask(seq_len, [0]),
        'Random (k=5)': create_random_mask(seq_len, num_random=5),
        'Strided (步长=4)': create_strided_mask(seq_len, stride=4),
        'Longformer': create_longformer_mask(seq_len, window_size=3, global_indices=[0]),
        'BigBird': create_bigbird_mask(seq_len, window_size=3, num_random=3, global_indices=[0])
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, mask) in enumerate(patterns.items()):
        ax = axes[idx]

        # 绘制mask
        ax.imshow(mask, cmap='Blues', interpolation='nearest')
        ax.set_title(f'{name}\n稀疏度: {np.sum(mask)/(seq_len*seq_len):.2%}', fontsize=12)
        ax.set_xlabel('Key位置')
        ax.set_ylabel('Query位置')

        # 添加网格
        ax.set_xticks(np.arange(0, seq_len, seq_len//8))
        ax.set_yticks(np.arange(0, seq_len, seq_len//8))
        ax.grid(False)

    plt.tight_layout()
    plt.savefig('/tmp/sparse_attention_patterns.png', dpi=150, bbox_inches='tight')
    print("稀疏模式可视化已保存到: /tmp/sparse_attention_patterns.png")
    plt.close()


def compare_sparsity_performance(seq_len=128, embed_dim=64):
    """
    比较不同稀疏度的性能

    Args:
        seq_len: 序列长度
        embed_dim: 嵌入维度
    """
    import time

    print("=" * 80)
    print("稀疏注意力性能对比")
    print("=" * 80)

    x = np.random.randn(seq_len, embed_dim)

    # 标准注意力（全连接）
    full_mask = np.ones((seq_len, seq_len))
    scores = np.random.randn(seq_len, seq_len)
    start = time.time()
    scores_masked = np.where(full_mask == 1, scores, -1e9)
    attn = softmax(scores_masked, axis=-1)
    full_time = time.time() - start

    print(f"\n{'模式':<20} {'稀疏度':<12} {'时间(ms)':<12} {'加速比':<12}")
    print("-" * 80)

    patterns = [
        ('Full (标准)', {'pattern': 'full'}),
        ('Local (窗口=3)', {'pattern': 'local', 'window_size': 3}),
        ('Local (窗口=5)', {'pattern': 'local', 'window_size': 5}),
        ('Random (k=5)', {'pattern': 'random', 'num_random': 5}),
        ('Longformer', {'pattern': 'longformer', 'window_size': 3, 'global_indices': [0]}),
        ('BigBird', {'pattern': 'bigbird', 'window_size': 3, 'num_random': 3, 'global_indices': [0]})
    ]

    for name, config in patterns:
        if config['pattern'] == 'full':
            sparsity = 1.0
            elapsed = full_time
            speedup = 1.0
        else:
            sparse_attn = SparseAttention(embed_dim, **config)
            start = time.time()
            _, mask, sparsity = sparse_attn.forward(x)
            elapsed = time.time() - start
            speedup = full_time / elapsed

        print(f"{name:<20} {sparsity:<12.2%} {elapsed*1000:<12.4f} {speedup:<12.2f}x")

    print("-" * 80)


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("稀疏注意力机制演示")
    print("=" * 80)

    # 参数设置
    seq_len = 32
    embed_dim = 64

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 1. 局部注意力
    print("\n1. 局部注意力（窗口大小=3）")
    print("-" * 80)
    local_attn = SparseAttention(embed_dim, pattern='local', window_size=3)
    output_local, mask_local, sparsity_local = local_attn.forward(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_local.shape}")
    print(f"稀疏度: {sparsity_local:.2%}")
    print(f"活跃连接数: {np.sum(mask_local)} / {seq_len * seq_len}")

    # 2. Longformer风格
    print("\n2. Longformer风格（局部+全局）")
    print("-" * 80)
    longformer_attn = SparseAttention(
        embed_dim,
        pattern='longformer',
        window_size=3,
        global_indices=[0, seq_len-1]  # 首尾为全局token
    )
    output_lf, mask_lf, sparsity_lf = longformer_attn.forward(x)

    print(f"输出形状: {output_lf.shape}")
    print(f"稀疏度: {sparsity_lf:.2%}")
    print(f"全局token位置: [0, {seq_len-1}]")

    # 3. BigBird风格
    print("\n3. BigBird风格（局部+全局+随机）")
    print("-" * 80)
    bigbird_attn = SparseAttention(
        embed_dim,
        pattern='bigbird',
        window_size=3,
        num_random=3,
        global_indices=[0]
    )
    output_bb, mask_bb, sparsity_bb = bigbird_attn.forward(x)

    print(f"输出形状: {output_bb.shape}")
    print(f"稀疏度: {sparsity_bb:.2%}")
    print(f"窗口大小: 3")
    print(f"随机连接: 3")

    # 4. 稀疏模式可视化
    print("\n4. 稀疏模式可视化")
    print("-" * 80)
    visualize_sparse_patterns(seq_len=64)

    # 5. 性能对比
    print("\n5. 性能对比")
    compare_sparsity_performance(seq_len=128, embed_dim=64)

    # 6. 不同序列长度的稀疏度分析
    print("\n6. 不同序列长度的稀疏度分析")
    print("-" * 80)
    print(f"{'序列长度':<12} {'Local(w=3)':<15} {'Longformer':<15} {'BigBird':<15}")
    print("-" * 80)

    for length in [64, 128, 256, 512, 1024]:
        local_mask = create_local_mask(length, window_size=3)
        lf_mask = create_longformer_mask(length, window_size=3, global_indices=[0])
        bb_mask = create_bigbird_mask(length, window_size=3, num_random=3, global_indices=[0])

        local_sparsity = np.sum(local_mask) / (length * length)
        lf_sparsity = np.sum(lf_mask) / (length * length)
        bb_sparsity = np.sum(bb_mask) / (length * length)

        print(f"{length:<12} {local_sparsity:<15.4%} {lf_sparsity:<15.4%} {bb_sparsity:<15.4%}")

    print("\n" + "=" * 80)
    print("稀疏注意力的关键优势:")
    print("=" * 80)
    print("✓ 显著降低计算复杂度 O(n²) → O(n×k)")
    print("✓ 减少内存占用")
    print("✓ 适合处理超长序列")
    print("✓ 通过精心设计的模式保持性能")
    print("\n稀疏模式选择:")
    print("- Local: 适合局部相关性强的任务（如语言建模）")
    print("- Longformer: 需要全局token的任务（如分类）")
    print("- BigBird: 平衡局部、全局和长距离依赖")
    print("=" * 80)
