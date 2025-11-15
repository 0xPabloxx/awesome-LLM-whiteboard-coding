"""
Flash Attention (闪存注意力) 实现

Flash Attention通过优化GPU内存访问模式来加速注意力计算，而不改变数学结果。
核心思想是使用分块（tiling）技术，减少对高带宽内存（HBM）的访问，充分利用快速的片上内存（SRAM）。

核心思想：
1. 标准注意力需要多次读写HBM（慢）
2. Flash Attention通过分块计算，减少HBM访问
3. 数学结果完全相同，但速度更快、内存更少

内存层次：
- SRAM (片上): 快但小（~20MB）
- HBM (显存): 慢但大（~40GB）
- Flash Attention目标：最大化SRAM利用，最小化HBM访问

优势：
- 速度提升2-4倍
- 内存占用减少（不需要存储完整的注意力矩阵）
- 数学上完全等价于标准注意力
- 支持长序列

应用：GPT-4、Llama、Claude等现代LLM都使用Flash Attention
"""

import numpy as np
import time


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class StandardAttention:
    """
    标准注意力实现（用于对比）

    内存访问模式：
    1. 从HBM读取Q、K、V
    2. 计算QK^T，写入HBM（n×n矩阵）
    3. 从HBM读取QK^T，计算Softmax，写回HBM
    4. 从HBM读取注意力矩阵和V，计算输出
    总共：大量的HBM读写操作
    """

    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def forward(self, x, return_stats=False):
        """
        标准注意力前向传播

        Args:
            x: 输入，形状 (seq_len, embed_dim)
            return_stats: 是否返回内存访问统计

        Returns:
            output: 输出，形状 (seq_len, embed_dim)
            stats: (可选) 内存访问统计
        """
        seq_len, embed_dim = x.shape

        # 投影
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 计算注意力得分（需要存储n×n矩阵）
        scores = np.dot(Q, K.T) / np.sqrt(embed_dim)

        # Softmax
        attention = softmax(scores, axis=-1)

        # 加权求和
        output = np.dot(attention, V)

        if return_stats:
            # 统计内存访问
            stats = {
                'attention_matrix_size': seq_len * seq_len * 4,  # float32
                'hbm_reads': seq_len * embed_dim * 3 + seq_len * seq_len * 2,
                'hbm_writes': seq_len * seq_len + seq_len * embed_dim
            }
            return output, stats

        return output


class FlashAttention:
    """
    Flash Attention实现

    核心技术：
    1. 分块（Tiling）：将Q、K、V分成小块
    2. 在线Softmax：增量计算Softmax，避免存储完整注意力矩阵
    3. 重计算：前向时不保存注意力矩阵，反向时重新计算

    内存访问模式：
    1. 分块加载Q、K、V到SRAM
    2. 在SRAM中计算注意力
    3. 增量更新输出，无需存储完整注意力矩阵
    总共：显著减少HBM访问
    """

    def __init__(self, embed_dim, block_size=64):
        """
        初始化Flash Attention

        Args:
            embed_dim: 嵌入维度
            block_size: 分块大小（模拟SRAM容量限制）
        """
        self.embed_dim = embed_dim
        self.block_size = block_size

        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def _online_softmax_update(self, m_old, l_old, m_new, l_new):
        """
        在线Softmax更新

        当看到新的最大值时，需要重新缩放之前的结果

        Args:
            m_old: 旧的最大值
            l_old: 旧的归一化因子
            m_new: 新的最大值
            l_new: 新的归一化因子

        Returns:
            m_global: 全局最大值
            l_global: 全局归一化因子
            correction: 修正因子
        """
        m_global = np.maximum(m_old, m_new)

        # 计算修正因子
        correction_old = np.exp(m_old - m_global)
        correction_new = np.exp(m_new - m_global)

        # 更新归一化因子
        l_global = correction_old * l_old + correction_new * l_new

        return m_global, l_global, correction_old

    def forward(self, x, return_stats=False):
        """
        Flash Attention前向传播

        使用分块计算，避免存储完整的n×n注意力矩阵

        Args:
            x: 输入，形状 (seq_len, embed_dim)
            return_stats: 是否返回内存访问统计

        Returns:
            output: 输出，形状 (seq_len, embed_dim)
            stats: (可选) 内存访问统计
        """
        seq_len, embed_dim = x.shape
        block_size = self.block_size

        # 投影
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 初始化输出和统计量
        output = np.zeros((seq_len, embed_dim))
        m = np.full(seq_len, -np.inf)  # 每行的最大值
        l = np.zeros(seq_len)  # 每行的归一化因子

        # 计算需要的块数
        num_blocks = (seq_len + block_size - 1) // block_size

        hbm_reads = 0
        hbm_writes = 0

        # 外循环：遍历Q的块
        for i in range(num_blocks):
            # Q块的范围
            q_start = i * block_size
            q_end = min((i + 1) * block_size, seq_len)
            Q_block = Q[q_start:q_end]  # (block_size, embed_dim)

            # 当前块的累积输出
            O_block = np.zeros((q_end - q_start, embed_dim))
            m_block = np.full(q_end - q_start, -np.inf)
            l_block = np.zeros(q_end - q_start)

            hbm_reads += (q_end - q_start) * embed_dim

            # 内循环：遍历K、V的块
            for j in range(num_blocks):
                # K、V块的范围
                kv_start = j * block_size
                kv_end = min((j + 1) * block_size, seq_len)
                K_block = K[kv_start:kv_end]  # (block_size, embed_dim)
                V_block = V[kv_start:kv_end]  # (block_size, embed_dim)

                hbm_reads += 2 * (kv_end - kv_start) * embed_dim

                # 计算当前块的注意力得分
                # (q_block_size, embed_dim) @ (embed_dim, kv_block_size)
                scores_block = np.dot(Q_block, K_block.T) / np.sqrt(embed_dim)

                # 计算当前块的最大值和exp
                m_new = np.max(scores_block, axis=1)
                scores_block_shifted = scores_block - m_new[:, None]
                exp_scores = np.exp(scores_block_shifted)
                l_new = np.sum(exp_scores, axis=1)

                # 在线Softmax更新
                m_global, l_global, correction = self._online_softmax_update(
                    m_block, l_block, m_new, l_new
                )

                # 更新累积输出
                # 旧输出需要重新缩放
                O_block = O_block * (correction * l_block / l_global)[:, None]

                # 加上新的贡献
                attention_block = exp_scores / l_global[:, None]
                O_block += np.dot(attention_block, V_block)

                # 更新统计量
                m_block = m_global
                l_block = l_global

            # 将块的输出写回全局
            output[q_start:q_end] = O_block
            m[q_start:q_end] = m_block
            l[q_start:q_end] = l_block

            hbm_writes += (q_end - q_start) * embed_dim

        if return_stats:
            stats = {
                'attention_matrix_size': 0,  # 不需要存储完整注意力矩阵
                'max_block_size': block_size * block_size * 4,
                'hbm_reads': hbm_reads,
                'hbm_writes': hbm_writes
            }
            return output, stats

        return output


def compare_implementations(seq_len=256, embed_dim=64, block_size=64):
    """
    比较标准注意力和Flash Attention的性能

    Args:
        seq_len: 序列长度
        embed_dim: 嵌入维度
        block_size: Flash Attention的块大小
    """
    print("=" * 80)
    print("Flash Attention vs 标准注意力")
    print("=" * 80)
    print(f"序列长度: {seq_len}, 嵌入维度: {embed_dim}, 块大小: {block_size}\n")

    # 生成输入
    x = np.random.randn(seq_len, embed_dim)

    # 标准注意力
    print("1. 标准注意力")
    print("-" * 80)
    std_attn = StandardAttention(embed_dim)
    start = time.time()
    output_std, stats_std = std_attn.forward(x, return_stats=True)
    time_std = time.time() - start

    print(f"计算时间: {time_std*1000:.4f} ms")
    print(f"注意力矩阵大小: {stats_std['attention_matrix_size'] / (1024*1024):.4f} MB")
    print(f"HBM读取: {stats_std['hbm_reads']:,} 元素")
    print(f"HBM写入: {stats_std['hbm_writes']:,} 元素")

    # Flash Attention
    print("\n2. Flash Attention")
    print("-" * 80)
    flash_attn = FlashAttention(embed_dim, block_size=block_size)
    start = time.time()
    output_flash, stats_flash = flash_attn.forward(x, return_stats=True)
    time_flash = time.time() - start

    print(f"计算时间: {time_flash*1000:.4f} ms")
    print(f"最大块大小: {stats_flash['max_block_size'] / (1024*1024):.4f} MB")
    print(f"HBM读取: {stats_flash['hbm_reads']:,} 元素")
    print(f"HBM写入: {stats_flash['hbm_writes']:,} 元素")

    # 对比
    print("\n3. 性能对比")
    print("-" * 80)
    speedup = time_std / time_flash
    memory_saving = (stats_std['attention_matrix_size'] - stats_flash['max_block_size']) / stats_std['attention_matrix_size']
    hbm_reduction = 1 - (stats_flash['hbm_reads'] + stats_flash['hbm_writes']) / (stats_std['hbm_reads'] + stats_std['hbm_writes'])

    print(f"速度提升: {speedup:.2f}x")
    print(f"内存节省: {memory_saving:.1%}")
    print(f"HBM访问减少: {hbm_reduction:.1%}")

    # 验证正确性
    print("\n4. 正确性验证")
    print("-" * 80)
    diff = np.abs(output_std - output_flash)
    print(f"平均绝对误差: {np.mean(diff):.2e}")
    print(f"最大绝对误差: {np.max(diff):.2e}")
    print(f"相对误差: {np.mean(diff) / np.mean(np.abs(output_std)):.2e}")

    if np.allclose(output_std, output_flash, rtol=1e-5, atol=1e-5):
        print("✓ 输出完全一致（在数值误差范围内）")
    else:
        print("✗ 输出存在差异")

    return output_std, output_flash


def benchmark_sequence_lengths(embed_dim=64, block_size=64):
    """
    测试不同序列长度的性能

    Args:
        embed_dim: 嵌入维度
        block_size: 块大小
    """
    print("\n" + "=" * 80)
    print("不同序列长度的性能对比")
    print("=" * 80)

    seq_lengths = [64, 128, 256, 512, 1024]
    print(f"\n{'序列长度':<12} {'标准(ms)':<15} {'Flash(ms)':<15} {'加速比':<12} {'内存节省':<12}")
    print("-" * 80)

    for seq_len in seq_lengths:
        x = np.random.randn(seq_len, embed_dim)

        # 标准注意力
        std_attn = StandardAttention(embed_dim)
        start = time.time()
        output_std, stats_std = std_attn.forward(x, return_stats=True)
        time_std = time.time() - start

        # Flash Attention
        flash_attn = FlashAttention(embed_dim, block_size=block_size)
        start = time.time()
        output_flash, stats_flash = flash_attn.forward(x, return_stats=True)
        time_flash = time.time() - start

        speedup = time_std / time_flash
        memory_saving = (stats_std['attention_matrix_size'] - stats_flash['max_block_size']) / stats_std['attention_matrix_size']

        print(f"{seq_len:<12} {time_std*1000:<15.4f} {time_flash*1000:<15.4f} {speedup:<12.2f}x {memory_saving:<12.1%}")

    print("-" * 80)


def benchmark_block_sizes(seq_len=512, embed_dim=64):
    """
    测试不同块大小的影响

    Args:
        seq_len: 序列长度
        embed_dim: 嵌入维度
    """
    print("\n" + "=" * 80)
    print("不同块大小的性能对比")
    print("=" * 80)

    block_sizes = [16, 32, 64, 128, 256]
    print(f"\n{'块大小':<12} {'时间(ms)':<15} {'HBM访问':<15} {'块内存(KB)':<15}")
    print("-" * 80)

    x = np.random.randn(seq_len, embed_dim)

    for block_size in block_sizes:
        flash_attn = FlashAttention(embed_dim, block_size=block_size)
        start = time.time()
        output, stats = flash_attn.forward(x, return_stats=True)
        elapsed = time.time() - start

        total_hbm = stats['hbm_reads'] + stats['hbm_writes']
        block_mem = stats['max_block_size'] / 1024  # KB

        print(f"{block_size:<12} {elapsed*1000:<15.4f} {total_hbm:<15,} {block_mem:<15.2f}")

    print("-" * 80)


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("Flash Attention 演示")
    print("=" * 80)

    # 基本对比
    compare_implementations(seq_len=256, embed_dim=64, block_size=64)

    # 不同序列长度
    benchmark_sequence_lengths(embed_dim=64, block_size=64)

    # 不同块大小
    benchmark_block_sizes(seq_len=512, embed_dim=64)

    print("\n" + "=" * 80)
    print("Flash Attention的核心优势:")
    print("=" * 80)
    print("✓ 速度提升2-4倍（减少HBM访问）")
    print("✓ 内存占用显著降低（不存储n×n注意力矩阵）")
    print("✓ 数学上完全等价（无近似）")
    print("✓ 支持更长序列")
    print("✓ 现代LLM的标配（GPT-4、Llama等）")
    print("\n核心技术:")
    print("- 分块计算（Tiling）")
    print("- 在线Softmax（增量更新）")
    print("- 重计算策略（前向不保存中间结果）")
    print("- 优化内存访问模式（SRAM优先）")
    print("=" * 80)
