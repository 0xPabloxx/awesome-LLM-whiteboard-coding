"""
Hard Attention (硬注意力) 实现

硬注意力只选择部分位置进行关注，是一种随机采样的注意力机制。
由于是离散选择，不可微分，需要使用强化学习（如REINFORCE算法）进行训练。

核心思想：
1. 计算每个位置的注意力概率
2. 根据概率分布随机采样一个位置（或top-k个位置）
3. 只使用选中位置的值

与Soft Attention的区别：
- Soft: 对所有位置加权求和（连续、可微）
- Hard: 只选择部分位置（离散、不可微）
"""

import numpy as np


def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class HardAttention:
    """
    硬注意力机制实现类

    特点：
    - 只选择部分位置（通常是1个）
    - 不可微分，需要强化学习训练
    - 计算效率高，只处理选中的位置
    - 可以看作是一种随机注意力机制
    """

    def __init__(self, hidden_dim, sample_method='stochastic'):
        """
        初始化硬注意力层

        Args:
            hidden_dim: 隐藏层维度
            sample_method: 采样方法
                - 'stochastic': 随机采样（用于训练）
                - 'greedy': 贪婪选择（用于推理）
        """
        self.hidden_dim = hidden_dim
        self.sample_method = sample_method
        # 用于计算注意力得分的参数
        self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b = np.zeros((hidden_dim,))

    def compute_attention_probs(self, query, keys):
        """
        计算注意力概率分布

        Args:
            query: 查询向量，形状为 (hidden_dim,)
            keys: 键向量序列，形状为 (seq_len, hidden_dim)

        Returns:
            probs: 注意力概率，形状为 (seq_len,)
        """
        seq_len = keys.shape[0]

        # 计算注意力得分
        scores = []
        for i in range(seq_len):
            # 简单的点积注意力
            score = np.dot(query, keys[i])
            scores.append(score)

        scores = np.array(scores)

        # Softmax得到概率分布
        probs = softmax(scores)

        return probs

    def sample_location(self, probs):
        """
        根据概率分布采样位置

        Args:
            probs: 概率分布，形状为 (seq_len,)

        Returns:
            location: 采样得到的位置索引
        """
        if self.sample_method == 'stochastic':
            # 随机采样（训练时使用）
            location = np.random.choice(len(probs), p=probs)
        else:  # greedy
            # 贪婪选择概率最大的位置（推理时使用）
            location = np.argmax(probs)

        return location

    def forward(self, query, keys, values, return_probs=False):
        """
        前向传播

        Args:
            query: 查询向量，形状为 (hidden_dim,)
            keys: 键向量序列，形状为 (seq_len, hidden_dim)
            values: 值向量序列，形状为 (seq_len, hidden_dim)
            return_probs: 是否返回注意力概率

        Returns:
            output: 输出向量，形状为 (hidden_dim,)
            location: 选择的位置索引
            (optional) probs: 注意力概率分布
        """
        # 计算注意力概率
        probs = self.compute_attention_probs(query, keys)

        # 采样位置
        location = self.sample_location(probs)

        # 只使用选中位置的值
        output = values[location]

        if return_probs:
            return output, location, probs
        else:
            return output, location


class HardAttentionTopK:
    """
    Top-K 硬注意力变体

    不是只选择1个位置，而是选择top-k个位置
    """

    def __init__(self, hidden_dim, k=3):
        """
        Args:
            hidden_dim: 隐藏层维度
            k: 选择的位置数量
        """
        self.hidden_dim = hidden_dim
        self.k = k

    def forward(self, query, keys, values):
        """
        前向传播

        Args:
            query: 查询向量，形状为 (hidden_dim,)
            keys: 键向量序列，形状为 (seq_len, hidden_dim)
            values: 值向量序列，形状为 (seq_len, hidden_dim)

        Returns:
            output: 输出向量，形状为 (hidden_dim,)
            locations: 选择的位置索引，形状为 (k,)
        """
        # 计算注意力得分
        scores = np.dot(keys, query)  # (seq_len,)

        # 选择top-k个位置
        locations = np.argsort(scores)[-self.k:]  # 取分数最高的k个

        # 对选中位置的值求平均
        output = np.mean(values[locations], axis=0)

        return output, locations


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)

    # 参数设置
    hidden_dim = 64
    seq_len = 10

    print("=" * 50)
    print("硬注意力机制演示")
    print("=" * 50)

    # 生成示例数据
    query = np.random.randn(hidden_dim)
    keys = np.random.randn(seq_len, hidden_dim)
    values = np.random.randn(seq_len, hidden_dim)

    # 测试随机采样硬注意力
    print("\n1. 随机采样硬注意力（训练模式）")
    attention_stochastic = HardAttention(hidden_dim, sample_method='stochastic')

    for i in range(3):
        output, location, probs = attention_stochastic.forward(
            query, keys, values, return_probs=True
        )
        print(f"\n第 {i+1} 次采样:")
        print(f"  选中位置: {location}")
        print(f"  该位置概率: {probs[location]:.4f}")
        print(f"  输出向量形状: {output.shape}")

    # 测试贪婪硬注意力
    print("\n" + "=" * 50)
    print("2. 贪婪硬注意力（推理模式）")
    attention_greedy = HardAttention(hidden_dim, sample_method='greedy')

    output, location, probs = attention_greedy.forward(
        query, keys, values, return_probs=True
    )
    print(f"\n选中位置: {location}")
    print(f"该位置概率: {probs[location]:.4f}")
    print(f"注意力概率分布（前5个）: {probs[:5]}")

    # 测试Top-K硬注意力
    print("\n" + "=" * 50)
    print("3. Top-K 硬注意力")
    k = 3
    attention_topk = HardAttentionTopK(hidden_dim, k=k)

    output, locations = attention_topk.forward(query, keys, values)
    print(f"\nK={k}")
    print(f"选中的位置: {locations}")
    print(f"输出向量形状: {output.shape}")

    print("\n" + "=" * 50)
    print("硬注意力 vs 软注意力对比")
    print("=" * 50)
    print("硬注意力特点：")
    print("  ✓ 只选择部分位置，计算效率高")
    print("  ✓ 可以实现真正的\"注意力聚焦\"")
    print("  ✗ 不可微分，需要强化学习训练")
    print("  ✗ 训练不稳定，方差较大")
    print("\n软注意力特点：")
    print("  ✓ 可微分，训练稳定")
    print("  ✓ 考虑所有位置的信息")
    print("  ✗ 计算开销大")
    print("  ✗ 注意力可能过于分散")
