"""
Ring Attention (环形注意力) 实现

Ring Attention是一种分布式注意力计算方法，通过在多个设备间环形传递K和V来计算超长序列的注意力。
由Liu等人在2023年提出，是突破单设备内存限制、处理极长序列的关键技术。

核心思想：
1. 将序列分块到多个设备上
2. 使用环形通信模式传递K和V块
3. 每个设备累积计算注意力输出
4. 突破单设备内存限制

关键特性：
- 序列长度不受单设备内存限制
- 通信与计算重叠
- 支持数百万token的序列
- 内存效率高

公式：
将序列分成D个块，每个设备处理一个块：
  Attention(Q_i) = Σ_{j=0}^{D-1} softmax(Q_i·K_j^T) · V_j

通过环形传递K_j和V_j，在D轮中计算完整的注意力。

应用场景：
- 超长序列处理（百万token级别）
- 长视频理解
- 基因组序列分析
- 需要极长上下文的任务
"""

import numpy as np
from typing import List, Tuple


def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class RingAttention:
    """
    Ring Attention (环形注意力)

    通过环形通信在多个设备间分布式计算注意力，
    突破单设备内存限制，支持超长序列。

    特点：
    - 分布式计算，不受单设备内存限制
    - 环形通信模式，通信高效
    - 支持数百万token的序列
    - 计算与通信可以重叠
    """

    def __init__(self, embed_dim, num_heads, num_devices=4, use_bias=True):
        """
        初始化环形注意力

        Args:
            embed_dim: 嵌入维度（必须能被num_heads整除）
            num_heads: 注意力头的数量
            num_devices: 设备数量（模拟分布式环境）
            use_bias: 是否使用偏置
        """
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_devices = num_devices
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

    def split_sequence_to_devices(self, x):
        """
        将序列分割到多个设备

        Args:
            x: 输入序列，形状为 (seq_len, ...)

        Returns:
            chunks: 列表，每个元素是一个设备上的数据块
        """
        seq_len = x.shape[0]
        chunk_size = seq_len // self.num_devices

        # 如果不能整除，最后一个设备处理剩余的
        chunks = []
        for i in range(self.num_devices):
            start = i * chunk_size
            if i == self.num_devices - 1:
                end = seq_len
            else:
                end = (i + 1) * chunk_size
            chunks.append(x[start:end])

        return chunks

    def ring_attention_single_head(self, Q, K, V, mask=None):
        """
        计算单个头的环形注意力

        Args:
            Q: Query，形状为 (seq_len, head_dim)
            K: Key，形状为 (seq_len, head_dim)
            V: Value，形状为 (seq_len, head_dim)
            mask: 注意力mask（可选）

        Returns:
            output: 输出，形状为 (seq_len, head_dim)
            attention_weights: 注意力权重（用于可视化）

        说明：
            这是环形注意力的核心算法：
            1. 将Q, K, V分块到num_devices个设备
            2. 每个设备保持自己的Q块不变
            3. K和V块在设备间环形传递
            4. 每轮计算Q与当前K/V块的注意力，累积结果
            5. num_devices轮后得到完整的注意力输出
        """
        seq_len = Q.shape[0]

        # 1. 分块到设备
        Q_chunks = self.split_sequence_to_devices(Q)
        K_chunks = self.split_sequence_to_devices(K)
        V_chunks = self.split_sequence_to_devices(V)

        # 2. 初始化每个设备的输出和归一化因子
        outputs = [np.zeros_like(Q_chunk) for Q_chunk in Q_chunks]
        max_scores = [np.full((Q_chunk.shape[0],), -np.inf)
                      for Q_chunk in Q_chunks]
        sum_exp = [np.zeros(Q_chunk.shape[0]) for Q_chunk in Q_chunks]

        # 用于记录完整的注意力权重（仅用于演示和可视化）
        attention_weights_full = np.zeros((seq_len, seq_len))

        # 3. 环形传递：num_devices轮
        for ring_step in range(self.num_devices):
            # 每个设备处理当前的K和V块
            for device_id in range(self.num_devices):
                # 计算当前轮次该设备应该处理哪个K/V块
                # 环形传递：每轮K/V向下一个设备传递
                kv_source_device = (device_id + ring_step) % self.num_devices

                Q_i = Q_chunks[device_id]
                K_j = K_chunks[kv_source_device]
                V_j = V_chunks[kv_source_device]

                # 计算注意力得分
                scores = np.dot(Q_i, K_j.T) / np.sqrt(self.head_dim)
                # scores: (chunk_size_q, chunk_size_k)

                # 应用mask（如果有）
                if mask is not None:
                    # 计算对应的全局索引
                    q_start = sum([Q_chunks[d].shape[0] for d in range(device_id)])
                    k_start = sum([K_chunks[d].shape[0] for d in range(kv_source_device)])
                    q_end = q_start + Q_i.shape[0]
                    k_end = k_start + K_j.shape[0]

                    mask_block = mask[q_start:q_end, k_start:k_end]
                    scores = np.where(mask_block == 0, -1e9, scores)

                # 在线Softmax：更新最大值和指数和
                # 这是数值稳定的在线softmax算法
                current_max = np.max(scores, axis=1)  # (chunk_size_q,)

                # 更新全局最大值
                new_max = np.maximum(max_scores[device_id], current_max)

                # 更新之前的累积值（需要重新缩放）
                scale_old = np.exp(max_scores[device_id] - new_max)
                outputs[device_id] *= scale_old[:, None]
                sum_exp[device_id] *= scale_old

                # 计算当前块的贡献
                scale_new = np.exp(current_max - new_max)
                exp_scores = np.exp(scores - current_max[:, None])  # (chunk_size_q, chunk_size_k)

                # 累积输出和指数和
                outputs[device_id] += scale_new[:, None] * np.dot(exp_scores, V_j)
                sum_exp[device_id] += scale_new * np.sum(exp_scores, axis=1)

                # 更新最大值
                max_scores[device_id] = new_max

                # 记录注意力权重（仅用于可视化）
                attention_weights_full[q_start:q_end, k_start:k_end] = exp_scores

        # 4. 最终归一化
        for device_id in range(self.num_devices):
            outputs[device_id] /= sum_exp[device_id][:, None]

        # 5. 合并所有设备的输出
        output = np.concatenate(outputs, axis=0)

        # 归一化注意力权重（用于可视化）
        attention_weights = attention_weights_full / attention_weights_full.sum(axis=1, keepdims=True)

        return output, attention_weights

    def forward(self, query, key=None, value=None, mask=None, return_attention=False):
        """
        前向传播

        Args:
            query: Query输入，形状为 (seq_len, embed_dim)
            key: Key输入。如果为None，则使用query（自注意力）
            value: Value输入。如果为None，则使用key
            mask: 注意力mask
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
        Q = self.split_heads(Q)  # (num_heads, seq_len, head_dim)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 对每个头计算环形注意力
        head_outputs = []
        attention_weights_list = []

        for h in range(self.num_heads):
            head_output, attn_weights = self.ring_attention_single_head(
                Q[h], K[h], V[h], mask=mask
            )
            head_outputs.append(head_output)
            attention_weights_list.append(attn_weights)

        # 4. 合并头
        multi_head_output = np.stack(head_outputs, axis=0)  # (num_heads, seq_len, head_dim)
        concatenated = self.combine_heads(multi_head_output)

        # 5. 输出投影
        output = np.dot(concatenated, self.W_o)
        if self.use_bias:
            output += self.b_o

        if return_attention:
            attention_weights = np.stack(attention_weights_list, axis=0)
            return output, attention_weights

        return output


class RingSelfAttention(RingAttention):
    """
    Ring Self-Attention (环形自注意力)

    是RingAttention的特例，Q、K、V都来自同一个输入。
    """

    def forward(self, x, mask=None, return_attention=False):
        """
        前向传播（自注意力版本）

        Args:
            x: 输入序列，形状为 (seq_len, embed_dim)
            mask: 注意力mask
            return_attention: 是否返回注意力权重

        Returns:
            output: 输出
            attention_weights: (可选) 注意力权重
        """
        return super().forward(x, x, x, mask=mask, return_attention=return_attention)


def create_causal_mask(seq_len):
    """
    创建因果mask（用于自回归模型）

    Args:
        seq_len: 序列长度

    Returns:
        mask: 下三角矩阵
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def analyze_ring_communication(seq_len, num_devices, embed_dim):
    """
    分析环形注意力的通信和内存

    Args:
        seq_len: 序列长度
        num_devices: 设备数量
        embed_dim: 嵌入维度
    """
    chunk_size = seq_len // num_devices

    print("=" * 80)
    print("环形注意力通信和内存分析")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  序列长度: {seq_len:,}")
    print(f"  设备数量: {num_devices}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  每个设备的块大小: {chunk_size:,}")

    # 单设备标准注意力的内存
    single_device_attn_memory = seq_len * seq_len * 4  # 4 bytes per float
    single_device_kv_memory = 2 * seq_len * embed_dim * 4
    single_device_total = single_device_attn_memory + single_device_kv_memory

    # Ring Attention的内存（每个设备）
    ring_attn_memory = chunk_size * chunk_size * 4
    ring_kv_memory = 2 * chunk_size * embed_dim * 4
    ring_total = ring_attn_memory + ring_kv_memory

    # 通信量
    kv_transfer_per_step = 2 * chunk_size * embed_dim * 4  # K和V
    total_communication = kv_transfer_per_step * (num_devices - 1)  # 不包括第一轮

    print(f"\n单设备标准注意力内存:")
    print(f"  注意力矩阵: {single_device_attn_memory / 1024 / 1024:.2f} MB")
    print(f"  K,V缓存: {single_device_kv_memory / 1024 / 1024:.2f} MB")
    print(f"  总计: {single_device_total / 1024 / 1024:.2f} MB")

    print(f"\nRing Attention每个设备内存:")
    print(f"  注意力矩阵: {ring_attn_memory / 1024 / 1024:.2f} MB")
    print(f"  K,V缓存: {ring_kv_memory / 1024 / 1024:.2f} MB")
    print(f"  总计: {ring_total / 1024 / 1024:.2f} MB")
    print(f"  内存节省: {(1 - ring_total/single_device_total)*100:.1f}%")

    print(f"\n通信量:")
    print(f"  每步传输: {kv_transfer_per_step / 1024 / 1024:.2f} MB")
    print(f"  总传输量: {total_communication / 1024 / 1024:.2f} MB")
    print(f"  传输轮次: {num_devices}")

    print(f"\n关键优势:")
    print(f"  ✓ 每个设备内存减少 {num_devices}x")
    print(f"  ✓ 支持的最大序列长度增加 {num_devices}x")
    print(f"  ✓ 计算与通信可以重叠")
    print(f"  ✓ 序列长度仅受总设备数限制")


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("Ring Attention (环形注意力) 演示")
    print("=" * 80)

    # 参数设置
    seq_len = 16  # 实际应用中可以是数百万
    embed_dim = 64
    num_heads = 4
    num_devices = 4  # 模拟4个设备

    print(f"\n配置:")
    print(f"  序列长度: {seq_len}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  设备数量: {num_devices}")
    print(f"  每个设备的块大小: {seq_len // num_devices}")

    # 1. 创建环形注意力层
    print("\n" + "=" * 80)
    print("1. 环形注意力计算")
    print("=" * 80)

    # 生成示例输入
    x = np.random.randn(seq_len, embed_dim)

    # 创建环形注意力层
    ring_attn = RingSelfAttention(embed_dim, num_heads, num_devices)

    # 前向传播
    output, attn_weights = ring_attn.forward(x, return_attention=True)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")

    print(f"\n工作原理:")
    print(f"  • 序列被分成{num_devices}块，每块{seq_len//num_devices}个token")
    print(f"  • 每块分配到一个设备")
    print(f"  • K和V块在设备间环形传递")
    print(f"  • 每个设备累积计算自己的Q块的注意力")
    print(f"  • {num_devices}轮后得到完整的注意力输出")

    # 2. 可视化设备分块
    print("\n" + "=" * 80)
    print("2. 设备分块模式")
    print("=" * 80)

    chunk_size = seq_len // num_devices
    print(f"\n设备分块（每个设备{chunk_size}个token）:")
    for device_id in range(num_devices):
        start = device_id * chunk_size
        end = start + chunk_size
        print(f"  设备{device_id}: token [{start:2d}:{end:2d})")

    # 3. 环形传递模拟
    print("\n" + "=" * 80)
    print("3. 环形传递过程")
    print("=" * 80)

    print(f"\n环形传递{num_devices}轮:")
    print(f"{'轮次':>6} | {'设备0':^15} | {'设备1':^15} | {'设备2':^15} | {'设备3':^15}")
    print("-" * 80)

    for ring_step in range(num_devices):
        row = f"{ring_step:>6} |"
        for device_id in range(num_devices):
            kv_source = (device_id + ring_step) % num_devices
            row += f" Q{device_id}×K{kv_source},V{kv_source} |"
        print(row)

    print("\n说明:")
    print("  • Q_i表示设备i的Query块（保持不变）")
    print("  • K_j,V_j表示设备j的Key/Value块（环形传递）")
    print("  • 每轮计算Q_i × K_j, V_j的注意力并累积")

    # 4. 内存和通信分析
    print("\n" + "=" * 80)
    print("4. 实际应用场景分析")
    print("=" * 80)

    # 模拟百万token序列
    analyze_ring_communication(
        seq_len=1_000_000,  # 100万token
        num_devices=8,
        embed_dim=4096
    )

    # 5. 对比标准注意力
    print("\n" + "=" * 80)
    print("5. Ring Attention vs 标准注意力")
    print("=" * 80)

    # 创建标准注意力作为对比（简化版）
    def standard_attention(Q, K, V):
        scores = np.dot(Q, K.T) / np.sqrt(Q.shape[-1])
        attn_weights = softmax(scores, axis=-1)
        output = np.dot(attn_weights, V)
        return output, attn_weights

    # 使用相同的Q, K, V进行对比
    Q_test = x @ ring_attn.W_q
    K_test = x @ ring_attn.W_k
    V_test = x @ ring_attn.W_v

    # 取第一个头进行对比
    Q_test_h0 = ring_attn.split_heads(Q_test)[0]
    K_test_h0 = ring_attn.split_heads(K_test)[0]
    V_test_h0 = ring_attn.split_heads(V_test)[0]

    output_standard, _ = standard_attention(Q_test_h0, K_test_h0, V_test_h0)
    output_ring, _ = ring_attn.ring_attention_single_head(Q_test_h0, K_test_h0, V_test_h0)

    # 计算差异
    diff = np.abs(output_standard - output_ring).mean()

    print(f"\n数值验证（第1个头）:")
    print(f"  标准注意力输出: {output_standard.shape}")
    print(f"  Ring注意力输出: {output_ring.shape}")
    print(f"  平均绝对差异: {diff:.6f}")

    if diff < 1e-5:
        print("  ✓ Ring Attention数值正确（与标准注意力一致）")
    else:
        print("  ⚠ 存在数值差异（可能由于浮点精度）")

    # 6. 因果mask示例
    print("\n" + "=" * 80)
    print("6. 带因果Mask的Ring Attention")
    print("=" * 80)

    causal_mask = create_causal_mask(seq_len)
    output_causal, attn_causal = ring_attn.forward(x, mask=causal_mask, return_attention=True)

    print(f"因果mask形状: {causal_mask.shape}")
    print(f"输出形状: {output_causal.shape}")
    print("\n用途:")
    print("  • 自回归语言模型")
    print("  • 大规模生成任务")
    print("  • 支持极长上下文的生成")

    # 7. 扩展性分析
    print("\n" + "=" * 80)
    print("7. 扩展性分析")
    print("=" * 80)

    print(f"\n不同设备数下的序列长度上限:")
    print(f"{'设备数':>8} {'单设备内存(MB)':>18} {'支持序列长度':>15} {'总序列长度':>15}")
    print("-" * 65)

    single_device_memory_mb = 24000  # 假设24GB显存
    embed_dim_test = 4096

    for num_dev in [1, 2, 4, 8, 16, 32]:
        # 估算单设备能支持的序列长度
        # 主要限制是注意力矩阵: chunk_size^2 * 4 bytes
        # 简化估算: chunk_size ≈ sqrt(memory_bytes / 4)
        chunk_size = int(np.sqrt(single_device_memory_mb * 1024 * 1024 / 4))
        total_seq_len = chunk_size * num_dev

        print(f"{num_dev:>8} {single_device_memory_mb:>18,} {chunk_size:>15,} {total_seq_len:>15,}")

    print("\n关键观察:")
    print("  ✓ 设备数翻倍，支持的序列长度翻倍")
    print("  ✓ 32个设备可以处理数百万token")
    print("  ✓ 只受总设备数限制，不受单设备内存限制")

    # 8. 实际应用场景
    print("\n" + "=" * 80)
    print("8. 实际应用场景")
    print("=" * 80)

    print("\nRing Attention的应用:")
    print("  • 超长文档理解（数十万到百万token）")
    print("  • 全书级别的文本分析")
    print("  • 长视频理解（数小时视频）")
    print("  • 基因组序列分析")
    print("  • 代码库级别的代码理解")
    print("  • 需要极长上下文的对话系统")

    print("\n技术优势:")
    print("  ✓ 突破单设备内存限制")
    print("  ✓ 线性扩展性（设备数）")
    print("  ✓ 计算与通信可以重叠")
    print("  ✓ 保持完整的注意力计算（非近似）")
    print("  ✓ 与Flash Attention等优化兼容")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("Ring Attention通过环形通信模式在多个设备间分布式计算注意力，")
    print("突破了单设备内存限制，使得处理百万级token的序列成为可能。")
    print("\n这是实现真正长上下文语言模型的关键技术！")
